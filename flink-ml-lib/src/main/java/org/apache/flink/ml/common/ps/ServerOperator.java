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

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.message.AllReduceM;
import org.apache.flink.ml.common.ps.message.InitializeModel;
import org.apache.flink.ml.common.ps.message.MessageType;
import org.apache.flink.ml.common.ps.message.MessageUtils;
import org.apache.flink.ml.common.ps.message.PullIndexM;
import org.apache.flink.ml.common.ps.message.PulledValueM;
import org.apache.flink.ml.common.ps.message.PushKvM;
import org.apache.flink.ml.common.ps.training.IterationStageList;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
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
 * The server operator maintains the shared parameters. It receives push/pull requests from {@link
 * WorkerOperator} and sends the answer request to {@link MirrorWorkerOperator}. It works closely
 * with {@link ModelUpdater} in the following way:
 *
 * <ul>
 *   <li>The server operator deals with the message from workers and decide when to process the
 *       received message.
 *   <li>The server operator calls {@link ModelUpdater#handlePush(long[], double[])} and {@link
 *       ModelUpdater#handlePull(long[])} to process the messages in detail.
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
 *
 * <p>TODO: Add support for maintaining multiple parameters on servers.
 */
public class ServerOperator extends AbstractStreamOperator<Tuple2<Integer, byte[]>>
        implements OneInputStreamOperator<Tuple2<Integer, byte[]>, Tuple2<Integer, byte[]>>,
                IterationListener<Tuple2<Integer, byte[]>> {
    /** Iteration stage list. */
    private final IterationStageList<?> iterationStageList;
    /** Number of workers to communicate with. */
    private final int numWorkers;
    /** The logic to answer push/pull request from workers. */
    private final ModelUpdater modelUpdater;
    /** Format of model data: start index, end index, dense double array. */
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
    /** The merger for all reduce requests. */
    private final AllReduceMerger allReduceMerger;
    /** The pending pull requests. */
    private ListState<byte[]> pendingPulls;

    public ServerOperator(
            IterationStageList<?> iterationStageList,
            int numWorkers,
            ModelUpdater modelUpdater,
            OutputTag<Tuple3<Long, Long, double[]>> modelOutputTag) {
        this.iterationStageList = iterationStageList;
        this.numWorkers = numWorkers;
        this.modelUpdater = modelUpdater;
        this.modelOutputTag = modelOutputTag;
        this.pushRequestMerger = new PushRequestMerger();
        this.allReduceMerger = new AllReduceMerger();
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
        MessageType type = MessageUtils.getMessageType(request);
        switch (type) {
            case PULL_INDEX:
                pendingPulls.add(request);
                break;
            case INITIALIZE_MODEL_AS_ZERO:
                InitializeModel initializeModelAsZeroM = InitializeModel.fromBytes(request);
                Preconditions.checkState(serverId == initializeModelAsZeroM.serverId);

                long start = initializeModelAsZeroM.startIndex;
                long end = initializeModelAsZeroM.endIndex;
                if (initializeModelAsZeroM.workerId == 0) {
                    modelUpdater.open(start, end);
                }
                break;
            case PUSH_KV:
                futuresInEpoch.add(
                        singleThreadExecutor.submit(
                                () -> pushRequestMerger.processPushRequest(request)));
                break;
            case ALL_REDUCE_VALUE:
                futuresInEpoch.add(
                        singleThreadExecutor.submit(
                                () -> allReduceMerger.processAllReduceRequest(request)));
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
            modelUpdater.handlePush(kvs.f0, kvs.f1);
        }

        Iterator<byte[]> pullsIterator = pendingPulls.get().iterator();
        if (pullsIterator.hasNext()) {
            // This is a pull stage.
            while (pullsIterator.hasNext()) {
                byte[] pull = pullsIterator.next();
                futuresInEpoch.add(singleThreadExecutor.submit(() -> processPullRequest(pull)));
            }
        }
        if (allReduceMerger.reducedResult != null) {
            // This is an all reduce stage.
            PulledValueM pulledValueM =
                    new PulledValueM(serverId, -1, allReduceMerger.reducedResult);
            for (int workerId = 0; workerId < numWorkers; workerId++) {
                int finalWorkerId = workerId;
                pulledValueM.workerId = finalWorkerId;
                futuresInEpoch.add(
                        singleThreadExecutor.submit(
                                () ->
                                        output.collect(
                                                new StreamRecord<>(
                                                        Tuple2.of(
                                                                finalWorkerId,
                                                                pulledValueM.toBytes())))));
            }
        }
        for (Future<?> future : futuresInEpoch) {
            future.get();
        }
        pendingPulls.clear();
        allReduceMerger.reducedResult = null;
        futuresInEpoch.clear();
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
        modelUpdater.initializeState(context);
        pushRequestMerger.initializeState(context);
        allReduceMerger.initializeState(context);
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
        allReduceMerger.snapshotState(context);
    }

    private Object processPullRequest(byte[] bytesData) {
        PullIndexM pullIndexM = PullIndexM.fromBytes(bytesData);
        Preconditions.checkState(serverId == pullIndexM.serverId);
        int workerId = pullIndexM.workerId;
        long[] indices = pullIndexM.indices;
        double[] pulledValues = modelUpdater.handlePull(indices);
        PulledValueM pulledValueM = new PulledValueM(serverId, workerId, pulledValues);
        StreamRecord<Tuple2<Integer, byte[]>> record =
                new StreamRecord<>(Tuple2.of(workerId, pulledValueM.toBytes()));

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

        private Object processPushRequest(byte[] pushKv) {
            PushKvM pushKvM = PushKvM.fromBytes(pushKv);
            Tuple2<long[], double[]> pushKvs = pushKvM.kvs;
            long[] keys = pushKvs.f0;
            double[] values = pushKvs.f1;

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
                Tuple2<long[], double[]> kvs =
                        MessageUtils.getLongDoubleArray(accumulatedKvsInBytes, 0);
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
                byte[] bytes = new byte[MessageUtils.getLongDoubleArraySizeInBytes(kvs)];
                MessageUtils.putLongDoubleArray(kvs, bytes, 0);
                accumulatedKvsState.add(bytes);
            }
        }
    }

    private static class AllReduceMerger implements Serializable {
        private double[] reducedResult;
        private ListState<double[]> reducedResultState;

        private void processAllReduceRequest(byte[] request) {
            AllReduceM allReduceM = AllReduceM.fromBytes(request);
            double[] receivedValues = allReduceM.values;
            if (reducedResult == null) {
                reducedResult = receivedValues;
            } else {
                Preconditions.checkArgument(reducedResult.length == receivedValues.length);
                reducedResult = allReduceM.aggregator.apply(receivedValues, reducedResult);
            }
        }

        private void initializeState(StateInitializationContext context) throws Exception {
            reducedResultState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<double[]>(
                                            "reducedResultState",
                                            PrimitiveArrayTypeInfo
                                                    .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
            reducedResult =
                    OperatorStateUtils.getUniqueElement(reducedResultState, "reducedResultState")
                            .orElse(null);
        }

        private void snapshotState(StateSnapshotContext context) throws Exception {
            reducedResultState.clear();
            if (reducedResult != null) {
                reducedResultState.add(reducedResult);
            }
        }
    }
}
