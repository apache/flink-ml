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
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.ml.common.ps.iterations.AllReduceStage;
import org.apache.flink.ml.common.ps.iterations.IterationStage;
import org.apache.flink.ml.common.ps.iterations.PullStage;
import org.apache.flink.ml.common.ps.iterations.PullStage.Aggregator;
import org.apache.flink.ml.common.ps.iterations.PushStage;
import org.apache.flink.ml.common.ps.iterations.ReduceScatterStage;
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

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * The server operator maintains the shared parameters. The shared parameters can be modeled as a
 * collection of {key:value} pairs. By default, the keys are evenly distributed across servers
 * through hash partitioning. For example, if there are two servers and the keys are {1,2,3,4,5,6},
 * then server-0 maintains keys {1,3,5} and server-1 maintains keys {2,4,6}.
 *
 * <p>The server receives push/pull/all-reduce/reduce-scatter requests from {@link WorkerOperator}
 * and sends the answer request to {@link WorkerOperator}. It works closely with {@link
 * ModelUpdater} in the following way:
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
 * <p>Moreover, it accepts all-reduce/reduce-scatter request from workers and returns the reduced
 * result to all workers. Note that the input of all-reduce/reduce-scatter operation is not going to
 * be used in {@link ModelUpdater}.
 *
 * @param <MT> output format of model data.
 */
public class ServerOperator<MT> extends AbstractStreamOperator<byte[]>
        implements OneInputStreamOperator<byte[], byte[]>, IterationListener<byte[]> {
    /** The iterationStage list. */
    private final List<IterationStage> stageList;
    /** Number of workers to communicate with. */
    private final int numWorkers;
    /** The logic to answer push/pull request from workers. */
    private final ModelUpdater<MT> modelUpdater;
    /** Output tag of model data. */
    private final OutputTag<MT> modelOutputTag;
    /** Index of the current server task. */
    private transient int serverId;
    /** Thread pool to answer push/pull requests, to decouple the network and computation. */
    private transient ExecutorService singleThreadExecutor;
    /** The future objects of thread calls in one epoch. */
    private transient List<Future<?>> futuresInEpoch;
    /**
     * The pending requests that server needs to send out responses (pull, all-reduce,
     * reduce-scatter).
     */
    private ListState<byte[]> pendingRequests;
    /**
     * The push request merged by stage id. We use map to store the merged push request since there
     * may be consecutive pushes.
     */
    private transient TreeMap<Integer, Long2ObjectOpenHashMap> accPushesByStage;

    private ListState<byte[]> accPushesByStageState;

    public ServerOperator(
            List<IterationStage> stageList,
            int numWorkers,
            ModelUpdater<MT> modelUpdater,
            OutputTag<MT> modelOutputTag) {
        this.stageList = stageList;
        this.numWorkers = numWorkers;
        this.modelUpdater = modelUpdater;
        this.modelOutputTag = modelOutputTag;
    }

    @Override
    public void open() throws Exception {
        super.open();
        this.serverId = getRuntimeContext().getIndexOfThisSubtask();
        this.singleThreadExecutor = Executors.newSingleThreadExecutor();
        this.futuresInEpoch = new ArrayList<>();
    }

    @Override
    public void processElement(StreamRecord<byte[]> element) throws Exception {
        Message message = new Message(element.getValue());
        IterationStage stage = stageList.get(message.getStageId() % stageList.size());
        if (stage instanceof PushStage) {
            futuresInEpoch.add(singleThreadExecutor.submit(() -> processPushRequest(message)));
        } else if (stage instanceof PullStage
                || stage instanceof AllReduceStage
                || stage instanceof ReduceScatterStage) {
            pendingRequests.add(message.bytes);
        } else {
            throw new IllegalStateException(
                    "Illegal iteration stage: " + stage.getClass().getSimpleName() + ".");
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<byte[]> collector) throws Exception {
        // Waits until the pushes are processed.
        for (Future<?> future : futuresInEpoch) {
            future.get();
        }
        futuresInEpoch.clear();
        // Uses the merged pushes to update model.
        for (Long2ObjectOpenHashMap currentAccPush : accPushesByStage.values()) {
            if (currentAccPush.size() > 0) {
                // The push is not empty.
                int numDoublesPerKey;
                Object object = currentAccPush.values().iterator().next();
                if (object instanceof Double) {
                    numDoublesPerKey = 1;
                } else {
                    numDoublesPerKey = ((double[]) object).length;
                }

                ObjectIterator<Map.Entry<Long, ?>> objectIterator =
                        currentAccPush.long2ObjectEntrySet().fastIterator();

                long[] assembledKeys = new long[currentAccPush.size()];
                double[] assembledValues = new double[currentAccPush.size() * numDoublesPerKey];

                int idx = 0;
                if (numDoublesPerKey == 1) {
                    while (objectIterator.hasNext()) {
                        Map.Entry<Long, Double> entry =
                                (Map.Entry<Long, Double>) objectIterator.next();
                        assembledKeys[idx] = entry.getKey();
                        assembledValues[idx] = entry.getValue();
                        idx++;
                    }
                } else {
                    while (objectIterator.hasNext()) {
                        Map.Entry<Long, double[]> entry =
                                (Map.Entry<Long, double[]>) objectIterator.next();
                        assembledKeys[idx] = entry.getKey();
                        System.arraycopy(
                                entry.getValue(),
                                0,
                                assembledValues,
                                idx * numDoublesPerKey,
                                numDoublesPerKey);
                        idx++;
                    }
                }
                currentAccPush.clear();
                modelUpdater.update(assembledKeys, assembledValues);
            }
        }

        // Deals with the pending requests, which should be one of Pull, AllReduce, ReduceScatter.
        Iterator<byte[]> requestIterator = pendingRequests.get().iterator();
        if (requestIterator.hasNext()) {
            Message message = new Message(requestIterator.next());
            int stageId = message.getStageId();
            IterationStage stage = stageList.get(stageId % stageList.size());
            requestIterator = pendingRequests.get().iterator();
            if (stage instanceof PullStage) {
                final int blockingQueueCapacity = 20;
                LinkedBlockingDeque<byte[]> pullsResponse =
                        new LinkedBlockingDeque<>(blockingQueueCapacity);
                for (byte[] bytes : pendingRequests.get()) {
                    singleThreadExecutor.submit(
                            () -> processPullRequest(new Message(bytes), pullsResponse));
                }
                int numResponsesSent = 0;
                while (numResponsesSent < numWorkers) {
                    Message response = new Message(pullsResponse.take());
                    output.collect(new StreamRecord<>(response.bytes));
                    numResponsesSent++;
                }
            } else if (stage instanceof AllReduceStage) {
                processAllReduceRequest(requestIterator);
            } else if (stage instanceof ReduceScatterStage) {
                processReduceScatterRequest(requestIterator);
            } else {
                throw new IllegalStateException(
                        "Illegal iteration stage: " + stage.getClass().getSimpleName() + ".");
            }

            pendingRequests.clear();
        }
    }

    @Override
    public void onIterationTerminated(Context context, Collector<byte[]> collector) {
        Iterator<MT> modelSegments = modelUpdater.getModelSegments();
        while (modelSegments.hasNext()) {
            MT modelSegment = modelSegments.next();
            output.collect(modelOutputTag, new StreamRecord<>(modelSegment));
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        pendingRequests =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "pendingRequests",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));

        modelUpdater.initializeState(context);

        accPushesByStageState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "accPushesByStageState",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));

        // Recovers accPushesByStage from a byte[] stream.
        Iterator<byte[]> accPushesInBytes = accPushesByStageState.get().iterator();
        accPushesByStage = new TreeMap<>();

        if (accPushesInBytes.hasNext()) {
            // 4 bytes for number of stages.
            byte[] meta = accPushesInBytes.next();
            int offset = 0;
            int numberOfStages = Bits.getInt(meta, offset);
            for (int i = 0; i < numberOfStages; i++) {
                byte[] oneStageMeta = accPushesInBytes.next();
                offset = 0;
                int stageId = Bits.getInt(oneStageMeta, offset);
                offset += Integer.BYTES;
                int sizeOfLong2ObjectMap = Bits.getInt(oneStageMeta, offset);
                offset += Integer.BYTES;
                int arrayLengthPerObject = Bits.getInt(oneStageMeta, offset);
                Long2ObjectOpenHashMap pushes;
                if (arrayLengthPerObject == 0) {
                    pushes = new Long2ObjectOpenHashMap<Double>(sizeOfLong2ObjectMap);
                } else {
                    pushes = new Long2ObjectOpenHashMap<double[]>(sizeOfLong2ObjectMap);
                }
                accPushesByStage.put(stageId, pushes);
                for (int entryId = 0; entryId < sizeOfLong2ObjectMap; entryId++) {
                    byte[] kvInBytes = accPushesInBytes.next();
                    long key = Bits.getLong(kvInBytes, 0);
                    if (arrayLengthPerObject == 0) {
                        Double value = Bits.getDouble(kvInBytes, Long.BYTES);
                        pushes.put(key, value);
                    } else {
                        double[] value = Bits.getDoubleArray(kvInBytes, Long.BYTES);
                        pushes.put(key, value);
                    }
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        // Waits until the futures to finish.
        for (Future<?> future : futuresInEpoch) {
            future.get();
        }
        futuresInEpoch.clear();
        modelUpdater.snapshotState(context);

        accPushesByStageState.clear();
        // Writes accPushesByStage to state in the following format:
        // numberOfStagesInInt,
        // stageIdInInt, sizeOfLong2ObjectMapInInt, arrayLengthPerObject, key-value-long-obj...
        // stageIdInInt, sizeOfLong2ObjectMapInInt, arrayLengthPerObject, key-value-long-obj...
        if (accPushesByStage.size() > 0) {
            int numberOfStages = accPushesByStage.size();
            byte[] meta = new byte[Integer.BYTES];
            Bits.putInt(meta, 0, numberOfStages);
            accPushesByStageState.add(meta);

            for (Map.Entry<Integer, Long2ObjectOpenHashMap> entry : accPushesByStage.entrySet()) {
                byte[] oneStageMeta = new byte[Integer.BYTES * 3];
                int offset = 0;
                int stageId = entry.getKey();
                Bits.putInt(oneStageMeta, offset, stageId);
                offset += Integer.BYTES;
                int sizeOfLong2ObjectMap = entry.getValue().size();
                Bits.putInt(oneStageMeta, offset, sizeOfLong2ObjectMap);
                offset += Integer.BYTES;
                // 0 stands for Double, a non-zero value represents the array length.
                int arrayLengthPerObject = 0;

                ObjectIterator<Map.Entry<Long, ?>> objectIterator =
                        entry.getValue().long2ObjectEntrySet().fastIterator();

                if (objectIterator.hasNext()) {
                    Map.Entry<Long, ?> oneEntry = objectIterator.next();
                    if (oneEntry.getValue() instanceof double[]) {
                        arrayLengthPerObject = ((double[]) (oneEntry.getValue())).length;
                    }
                    Bits.putInt(oneStageMeta, offset, arrayLengthPerObject);
                    accPushesByStageState.add(oneStageMeta);

                    accPushesByStageState.add(kvToBytes(oneEntry));
                    while (objectIterator.hasNext()) {
                        accPushesByStageState.add(kvToBytes(objectIterator.next()));
                    }
                }
            }
        }
    }

    private static byte[] kvToBytes(Map.Entry<Long, ?> obj) {
        byte[] bytes;
        if (obj.getValue() instanceof double[]) {
            double[] value = (double[]) obj.getValue();
            bytes = new byte[Long.BYTES + Bits.getDoubleArraySizeInBytes(value)];
            Bits.putLong(bytes, 0, obj.getKey());
            Bits.putDoubleArray(value, bytes, Long.BYTES);
        } else {
            bytes = new byte[Long.BYTES + Double.BYTES];
            Bits.putLong(bytes, 0, obj.getKey());
            Bits.putDouble(bytes, Long.BYTES, (Double) obj.getValue());
        }
        return bytes;
    }

    @SuppressWarnings("unchecked")
    private Object processPushRequest(Message message) throws Exception {
        long[] keys = message.getKeys();
        int stageId = message.getStageId();
        double[] values = message.getValuesInDoubleArray();

        accPushesByStage.putIfAbsent(stageId, new Long2ObjectOpenHashMap());
        Long2ObjectOpenHashMap currentAccKvs = accPushesByStage.get(stageId);

        if (keys.length != 0) {
            ReduceFunction<Double> reduceFunc =
                    ((PushStage) stageList.get(stageId % stageList.size())).reduceFunc;
            if (values.length == keys.length) {
                for (int i = 0; i < keys.length; i++) {
                    if (currentAccKvs.containsKey(keys[i])) {
                        double currentVal = (Double) currentAccKvs.get(keys[i]);
                        currentAccKvs.put(keys[i], reduceFunc.reduce(currentVal, values[i]));
                    } else {
                        currentAccKvs.put(keys[i], (Double) values[i]);
                    }
                }
            } else {
                int numDoublesPerKey = values.length / keys.length;
                for (int i = 0; i < keys.length; i++) {
                    if (currentAccKvs.containsKey(keys[i])) {
                        double[] currentVal = (double[]) currentAccKvs.get(keys[i]);
                        for (int j = 0; j < numDoublesPerKey; j++) {
                            currentVal[j] =
                                    reduceFunc.reduce(
                                            currentVal[j], values[i * numDoublesPerKey + j]);
                        }
                    } else {
                        currentAccKvs.put(
                                keys[i],
                                Arrays.copyOfRange(
                                        values,
                                        i * numDoublesPerKey,
                                        i * numDoublesPerKey + numDoublesPerKey));
                    }
                }
            }
        }
        return new Object();
    }

    private void processPullRequest(Message message, LinkedBlockingDeque<byte[]> pullsResponse) {
        int workerId = message.getWorkerId();
        long[] keys = message.getKeys();
        Message response;

        if (keys.length == 0) {
            // No request on this server.
            response =
                    new Message(
                            workerId, serverId, message.getStageId(), new long[0], new double[0]);
        } else {
            double[] pulledValues = modelUpdater.get(keys);
            Preconditions.checkState(pulledValues.length % keys.length == 0);
            int numDoublesPerKey = pulledValues.length / keys.length;

            double[] aggregatedPullValues = null;
            Aggregator<double[], double[]> aggregator =
                    ((PullStage) (stageList.get(message.getStageId() % stageList.size())))
                            .aggregator;
            if (aggregator != null) {
                // Processes the pulled values if the aggregator is not null.
                double[] tmp = new double[numDoublesPerKey];
                for (int i = 0; i < keys.length; i++) {
                    System.arraycopy(pulledValues, i * numDoublesPerKey, tmp, 0, numDoublesPerKey);
                    aggregatedPullValues = aggregator.add(tmp, aggregatedPullValues);
                }
            } else {
                aggregatedPullValues = pulledValues;
            }

            response =
                    new Message(
                            workerId,
                            serverId,
                            message.getStageId(),
                            new long[0],
                            aggregatedPullValues);
        }
        while (!pullsResponse.offer(response.bytes)) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private <V> void processAllReduceRequest(Iterator<byte[]> requests) throws Exception {
        byte[] request = requests.next();
        Message message = new Message(request);
        int stageId = message.getStageId();
        AllReduceStage<V> stage = (AllReduceStage<V>) stageList.get(stageId % stageList.size());
        V[] reducedResult = message.getValues(stage.typeSerializer);
        ReduceFunction<V[]> reduceFunction = stage.reducer;

        while (requests.hasNext()) {
            message = new Message(requests.next());
            reducedResult =
                    reduceFunction.reduce(message.getValues(stage.typeSerializer), reducedResult);
        }
        message =
                new Message(
                        -1, serverId, stageId, new long[0], reducedResult, stage.typeSerializer);

        for (int workerId = 0; workerId < numWorkers; workerId++) {
            message.setWorkerId(workerId);
            output.collect(new StreamRecord<>(message.bytes));
        }
    }

    @SuppressWarnings("unchecked")
    private <V> void processReduceScatterRequest(Iterator<byte[]> requests) throws Exception {
        byte[] request = requests.next();
        Message message = new Message(request);
        int stageId = message.getStageId();
        ReduceScatterStage<V> stage =
                (ReduceScatterStage<V>) stageList.get(stageId % stageList.size());
        V[] reducedResult = message.getValues(stage.typeSerializer);
        ReduceFunction<V[]> reduceFunction = stage.reducer;

        while (requests.hasNext()) {
            message = new Message(requests.next());
            reducedResult =
                    reduceFunction.reduce(message.getValues(stage.typeSerializer), reducedResult);
        }

        int[] recvCounts = stage.recvCounts;
        int totalCnt = Arrays.stream(recvCounts).sum();
        int shardSize = totalCnt / getRuntimeContext().getNumberOfParallelSubtasks() + 1;
        int sliceStart = Math.min(serverId * shardSize, totalCnt);
        int sliceEnd = Math.min(sliceStart + shardSize, totalCnt);

        int s = 0;
        int e;
        for (int workerId = 0; workerId < numWorkers; workerId++) {
            e = recvCounts[workerId] + s;

            int intersectionStart = Math.max(s, sliceStart);
            int interSectionEnd = Math.min(e, sliceEnd);
            int copyStart = 0, copyEnd = 0;
            if (interSectionEnd > intersectionStart) {
                copyStart = intersectionStart - sliceStart;
                copyEnd = interSectionEnd - sliceStart;
            }
            message =
                    new Message(
                            workerId,
                            serverId,
                            stageId,
                            new long[0],
                            Arrays.copyOfRange(reducedResult, copyStart, copyEnd),
                            stage.typeSerializer);
            output.collect(new StreamRecord<>(message.bytes));
        }
    }
}
