/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.message.Message;
import org.apache.flink.ml.common.ps.message.MessageType;
import org.apache.flink.ml.common.ps.training.AllReduceStage;
import org.apache.flink.ml.common.ps.training.IterationStage;
import org.apache.flink.ml.common.ps.training.IterationStageList;
import org.apache.flink.ml.common.ps.training.PullStage;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.ml.util.Bits;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The server operator maintains the shared parameters. It receives push/pull/allreduce requests
 * from {@link WorkerOperator} and sends the answer request to {@link ResponseAssemblerOperator}. It
 * works closely with {@link ModelUpdater} in the following way:
 *
 * <ul>
 *   <li>The server operator deals with the message from workers and decides when to process the
 *       received message.
 *   <li>The server operator calls {@link ModelUpdater#update(long[], double[])} and {@link
 *       ModelUpdater#get(long[])} to process the messages in detail.
 *   <li>The server operator triggers checkpoint for {@link ModelUpdater}.
 *   <li>The server operator outputs the final output parameters by calling {@link
 *       ModelUpdater#getModelSegments()}.
 * </ul>
 *
 * <p>Moreover, it accepts all-reduce request from workers and returns the reduced result to all
 * workers. Note that the input of all reduce operation is not going to be used in {@link
 * ModelUpdater}.
 *
 * <p>TODO: Add support for asynchronous operations on servers.
 */
public class ServerOperator extends AbstractStreamOperator<Tuple2<Integer, byte[]>>
        implements OneInputStreamOperator<Tuple2<Integer, byte[]>, Tuple2<Integer, byte[]>>,
                IterationListener<Tuple2<Integer, byte[]>> {
    /** The iterationStage list that asks responses from servers. */
    private final List<IterationStage> stageList;
    /** Number of workers to communicate with. */
    private final int numWorkers;
    /** The logic to answer push/pull request from workers. */
    private final ModelUpdater modelUpdater;
    /** Format of model data: start key index, end key index, dense double array. */
    private final OutputTag<Tuple3<Long, Long, double[]>> modelOutputTag;
    /** Index of the server task. */
    private int serverId = -1;
    /**
     * Thread pool to answer push/pull requests, to decouple the network traffic and computation
     * logic.
     */
    private transient ExecutorService singleThreadExecutor;
    /** The future objects of thread calls in one epoch. */
    private final List<Future<?>> futuresInEpoch = new ArrayList<>();
    /** The merger for push requests. */
    private final PushRequestMerger pushRequestMerger;
    /** The pending pull requests. */
    private ListState<byte[]> pendingPulls;

    /** The pending allreduce requests. */
    private ListState<byte[]> pendingAllReduces;

    public ServerOperator(
            IterationStageList<?> iterationStageList,
            int numWorkers,
            ModelUpdater modelUpdater,
            OutputTag<Tuple3<Long, Long, double[]>> modelOutputTag) {
        this.stageList = new ArrayList<>();
        for (IterationStage stage : iterationStageList.stageList) {
            if (stage instanceof PullStage || stage instanceof AllReduceStage) {
                stageList.add(stage);
            }
        }
        this.numWorkers = numWorkers;
        this.modelUpdater = modelUpdater;
        this.modelOutputTag = modelOutputTag;
        this.pushRequestMerger = new PushRequestMerger();
    }

    @Override
    public void open() throws Exception {
        super.open();
        this.serverId = getRuntimeContext().getIndexOfThisSubtask();
        this.singleThreadExecutor = Executors.newSingleThreadExecutor();
    }

    @Override
    public void processElement(StreamRecord<Tuple2<Integer, byte[]>> element) throws Exception {
        byte[] request = element.getValue().f1;
        Message message = new Message(element.getValue().f1);
        MessageType type = message.getMessageType();
        switch (type) {
            case INITIALIZE:
                long[] indices = message.getKeys();
                Preconditions.checkState(serverId == message.getServerId() && indices.length == 2);
                if (message.getWorkerId() == 0) {
                    modelUpdater.open(indices[0], indices[1]);
                }
                break;
            case PUSH:
                futuresInEpoch.add(
                        singleThreadExecutor.submit(
                                () -> pushRequestMerger.processPushRequest(message)));
                break;
            case PULL:
                pendingPulls.add(request);
                break;
            case ALL_REDUCE:
                pendingAllReduces.add(request);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported message type: " + type + ".");
        }
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Tuple2<Integer, byte[]>> collector)
            throws Exception {
        // Waits until all pushes have been processed.
        for (Future<?> future : futuresInEpoch) {
            future.get();
        }
        futuresInEpoch.clear();

        // Processes the pushes first.
        Tuple2<long[], double[]> kvs = pushRequestMerger.toKvArrays();
        pushRequestMerger.accumulatedKvsForMatrix.clear();
        pushRequestMerger.accumulatedKvsForVector.clear();
        if (kvs.f0.length > 0) {
            // There are pushes at this epoch.
            modelUpdater.update(kvs.f0, kvs.f1);
        }

        Iterator<byte[]> pullsIterator = pendingPulls.get().iterator();
        if (pullsIterator.hasNext()) {
            // This is a pull stage.
            while (pullsIterator.hasNext()) {
                byte[] pull = pullsIterator.next();
                futuresInEpoch.add(
                        singleThreadExecutor.submit(() -> processPullRequest(new Message(pull))));
            }
        }
        Iterator<byte[]> allreduceIterator = pendingAllReduces.get().iterator();
        if (allreduceIterator.hasNext()) {
            int stageId = epochWatermark % stageList.size();
            AllReduceStage<?> allReduceStage = (AllReduceStage<?>) stageList.get(stageId);
            Message reducedResult = processAllReduceRequest(allReduceStage, allreduceIterator);
            for (int workerId = 0; workerId < numWorkers; workerId++) {
                reducedResult.setWorkerId(workerId);
                output.collect(new StreamRecord<>(Tuple2.of(workerId, reducedResult.bytes)));
            }
        }

        for (Future<?> future : futuresInEpoch) {
            future.get();
        }
        pendingPulls.clear();
        pendingAllReduces.clear();
        futuresInEpoch.clear();
    }

    private <V> Message processAllReduceRequest(AllReduceStage<V> stage, Iterator<byte[]> requests)
            throws Exception {
        ReduceFunction<V[]> reduceFunction = stage.reducer;
        V[] reducedResult = null;
        while (requests.hasNext()) {
            byte[] allreduceRequest = requests.next();
            Message message = new Message(allreduceRequest);
            V[] receivedResult = message.getValues(stage.typeSerializer);
            if (reducedResult == null) {
                reducedResult = receivedResult;
            } else {
                reducedResult = reduceFunction.reduce(receivedResult, reducedResult);
            }
        }

        return new Message(
                -1, -1, MessageType.ALL_REDUCE, new long[0], reducedResult, stage.typeSerializer);
    }

    @Override
    public void onIterationTerminated(
            Context context, Collector<Tuple2<Integer, byte[]>> collector) {
        Iterator<Tuple3<Long, Long, double[]>> modelSegments = modelUpdater.getModelSegments();
        while (modelSegments.hasNext()) {
            Tuple3<Long, Long, double[]> modelSegment = modelSegments.next();
            output.collect(modelOutputTag, new StreamRecord<>(modelSegment));
        }
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        pendingPulls =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "pendingPulls",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));
        pendingAllReduces =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "pendingAllReduces",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));
        modelUpdater.initializeState(context);
        pushRequestMerger.initializeState(context);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);

        // Waits until the futures to finish.
        for (Future<?> future : futuresInEpoch) {
            future.get();
        }
        futuresInEpoch.clear();
        modelUpdater.snapshotState(context);
        pushRequestMerger.snapshotState(context);
    }

    private Object processPullRequest(Message message) {
        Preconditions.checkState(serverId == message.getServerId());
        int workerId = message.getWorkerId();
        double[] pulledValues = modelUpdater.get(message.getKeys());
        Message pulledMessage =
                new Message(serverId, workerId, MessageType.PULL, new long[0], pulledValues);
        StreamRecord<Tuple2<Integer, byte[]>> record =
                new StreamRecord<>(Tuple2.of(workerId, pulledMessage.bytes));

        output.collect(record);
        return new Object();
    }

    /** Utility class to merge the push request from different workers. */
    private static class PushRequestMerger implements Serializable {
        /** The accumulated kv if the push request is for a vector. */
        private final Long2DoubleOpenHashMap accumulatedKvsForVector;
        /** The accumulated kv if the push request is for a matrix. */
        private final Map<Long, double[]> accumulatedKvsForMatrix;
        /** The state for accumulated kv. */
        private ListState<byte[]> accumulatedKvsState;

        public PushRequestMerger() {
            this.accumulatedKvsForVector = new Long2DoubleOpenHashMap();
            this.accumulatedKvsForMatrix = new HashMap<>();
        }

        private Object processPushRequest(Message message) {
            long[] keys = message.getKeys();
            double[] values = message.getValuesInDoubleArray();

            if (values.length == keys.length) {
                for (int i = 0; i < keys.length; i++) {
                    accumulatedKvsForVector.merge(keys[i], values[i], Double::sum);
                }
            } else {
                int valuesPerKey = values.length / keys.length;
                for (int i = 0; i < keys.length; i++) {
                    accumulatedKvsForMatrix.putIfAbsent(keys[i], new double[valuesPerKey]);
                    double[] partialValue = accumulatedKvsForMatrix.get(keys[i]);
                    for (int j = 0; j < valuesPerKey; j++) {
                        partialValue[j] += values[i * valuesPerKey + j];
                    }
                }
            }
            return new Object();
        }

        /** Transforms the processed push request to kv arrays. */
        private Tuple2<long[], double[]> toKvArrays() {
            long[] indices = new long[0];
            double[] values = new double[0];
            if (accumulatedKvsForVector.size() != 0) {
                indices = new long[accumulatedKvsForVector.size()];
                values = new double[indices.length];

                int idx = 0;
                for (Map.Entry<Long, Double> entry : accumulatedKvsForVector.entrySet()) {
                    indices[idx] = entry.getKey();
                    values[idx] = entry.getValue();
                    idx++;
                }
            } else if (accumulatedKvsForMatrix.size() != 0) {
                indices = new long[accumulatedKvsForMatrix.size()];
                int numValuesPerKey =
                        accumulatedKvsForMatrix.entrySet().iterator().next().getValue().length;
                values = new double[indices.length * numValuesPerKey];
                int idx = 0;
                for (Map.Entry<Long, double[]> entry : accumulatedKvsForMatrix.entrySet()) {
                    indices[idx] = entry.getKey();
                    System.arraycopy(
                            entry.getValue(), 0, values, idx * numValuesPerKey, numValuesPerKey);
                    idx++;
                }
            }
            return Tuple2.of(indices, values);
        }

        private void initializeState(StateInitializationContext context) throws Exception {
            accumulatedKvsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "accumulatedKvs",
                                            PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));

            byte[] accumulatedKvsInBytes =
                    OperatorStateUtils.getUniqueElement(accumulatedKvsState, "accumulatedKvs")
                            .orElse(null);

            if (accumulatedKvsInBytes != null) {
                Tuple2<long[], double[]> kvs = Bits.getLongDoubleArray(accumulatedKvsInBytes, 0);
                long[] keys = kvs.f0;
                double[] values = kvs.f1;
                int numValuesPerKey = values.length / keys.length;
                if (numValuesPerKey == 1) {
                    for (int i = 0; i < keys.length; i++) {
                        accumulatedKvsForVector.put(keys[i], values[i]);
                    }
                } else {
                    for (int i = 0; i < keys.length; i++) {
                        accumulatedKvsForMatrix.put(
                                keys[i],
                                Arrays.copyOfRange(
                                        values,
                                        i * numValuesPerKey,
                                        i * numValuesPerKey + numValuesPerKey));
                    }
                }
            }
        }

        private void snapshotState(StateSnapshotContext context) throws Exception {
            Tuple2<long[], double[]> kvs = toKvArrays();
            accumulatedKvsState.clear();
            if (kvs.f0.length > 0) {
                byte[] bytes = new byte[Bits.getLongDoubleArraySizeInBytes(kvs)];
                Bits.putLongDoubleArray(kvs, bytes, 0);
                accumulatedKvsState.add(bytes);
            }
        }
    }
}
