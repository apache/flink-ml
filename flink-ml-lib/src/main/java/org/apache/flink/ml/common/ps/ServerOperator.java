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
import org.apache.flink.ml.common.ps.message.IndicesToPullM;
import org.apache.flink.ml.common.ps.message.KVsToPushM;
import org.apache.flink.ml.common.ps.message.MessageType;
import org.apache.flink.ml.common.ps.message.MessageUtils;
import org.apache.flink.ml.common.ps.message.ValuesPulledM;
import org.apache.flink.ml.common.ps.message.ZerosToPushM;
import org.apache.flink.ml.common.updater.ModelUpdater;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.SerializableObject;

import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
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
 *       received message. (i.e., synchronous vs. asynchronous).
 *   <li>The server operator calls {@link ModelUpdater#handlePush(long[], double[])} and {@link
 *       ModelUpdater#handlePull(long[])} to process the messages in detail.
 *   <li>The server operator ensures that {@link ModelUpdater} is robust to failures.
 *   <li>The server operator outputs the final output parameters by calling {@link
 *       ModelUpdater#getModelPieces()}.
 * </ul>
 *
 * <p>TODO: Add support for asynchronous operations on servers.
 *
 * <p>TODO: Add support for maintaining multiple parameters on servers.
 */
public class ServerOperator extends AbstractStreamOperator<Tuple2<Integer, byte[]>>
        implements OneInputStreamOperator<Tuple2<Integer, byte[]>, Tuple2<Integer, byte[]>>,
                IterationListener<Tuple2<Integer, byte[]>> {
    /** The logic to answer push/pull request from workers. */
    private final ModelUpdater modelUpdater;
    /** Format of model data: start index, end index, dense double array. */
    private final OutputTag<Tuple3<Long, Long, double[]>> modelOutputTag;

    private int serverId = -1;

    /**
     * Lock for output records to downstream operators. Note that we use multiple threads to deal
     * with push/pull requests for better performance.
     */
    private final SerializableObject lock = new SerializableObject();
    /** Number of threads to answer push/pull requests. */
    private final int numServerCores;
    /** Thread pool to answer push/pull requests. */
    private transient ExecutorService fixedThreadPool;
    /** The future objects of thread calls in one epoch. */
    private final List<Future<?>> futuresInEpoch = new ArrayList<>();
    /** The accumulated push request from workers by threadId. */
    private final ConcurrentHashMap<Long, Long2DoubleOpenHashMap> accumulatedKvsByThreadId;
    /** The accumulated results of Kvs. */
    private final Long2DoubleOpenHashMap accumulatedKvs;
    /** The state for accumulated Kvs. */
    private ListState<byte[]> accumulatedKvsState;
    /** The pending pull requests. */
    private ListState<byte[]> pendingPulls;

    public ServerOperator(
            ModelUpdater modelUpdater,
            OutputTag<Tuple3<Long, Long, double[]>> modelOutputTag,
            int numServerCores) {
        this.modelUpdater = modelUpdater;
        this.modelOutputTag = modelOutputTag;
        this.numServerCores = numServerCores;
        this.accumulatedKvsByThreadId = new ConcurrentHashMap<>();
        this.accumulatedKvs = new Long2DoubleOpenHashMap();
    }

    @Override
    public void open() throws Exception {
        super.open();
        serverId = getRuntimeContext().getIndexOfThisSubtask();
        fixedThreadPool = Executors.newFixedThreadPool(numServerCores);
    }

    @Override
    public void processElement(StreamRecord<Tuple2<Integer, byte[]>> element) throws Exception {
        byte[] request = element.getValue().f1;
        MessageType type = MessageUtils.getMessageType(request);
        if (type == MessageType.INDICES_TO_PULL) {
            pendingPulls.add(request);
        } else {
            processPushRequest(request);
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

        Iterator<Long2DoubleOpenHashMap> kvsFromAllThreads =
                accumulatedKvsByThreadId.values().iterator();
        if (kvsFromAllThreads.hasNext()) {
            Tuple2<long[], double[]> kvs = mergeKvsFromAllThreads(kvsFromAllThreads);
            modelUpdater.handlePush(kvs.f0, kvs.f1);
            accumulatedKvs.clear();
        }

        Iterator<byte[]> pullsIterator = pendingPulls.get().iterator();
        if (pullsIterator.hasNext()) {
            // The last iteration contains no pulls.
            while (pullsIterator.hasNext()) {
                byte[] pull = pullsIterator.next();
                futuresInEpoch.add(fixedThreadPool.submit(() -> processPullRequest(pull)));
            }
        }
        for (Future<?> future : futuresInEpoch) {
            future.get();
        }
        pendingPulls.clear();
        futuresInEpoch.clear();
    }

    @Override
    public void onIterationTerminated(
            Context context, Collector<Tuple2<Integer, byte[]>> collector) {
        Iterator<Tuple3<Long, Long, double[]>> modelPieces = modelUpdater.getModelPieces();
        while (modelPieces.hasNext()) {
            Tuple3<Long, Long, double[]> modelPiece = modelPieces.next();
            output.collect(modelOutputTag, new StreamRecord<>(modelPiece));
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
                    MessageUtils.readLongDoubleArray(accumulatedKvsInBytes, 0);
            accumulatedKvs.clear();
            for (int i = 0; i < kvs.f0.length; i++) {
                accumulatedKvs.put(kvs.f0[i], kvs.f1[i]);
            }
        }
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

        // Snapshots the pending pushes.
        Tuple2<long[], double[]> kvs =
                mergeKvsFromAllThreads(accumulatedKvsByThreadId.values().iterator());
        accumulatedKvsState.clear();
        if (kvs.f0.length > 0) {
            byte[] bytes = new byte[MessageUtils.getLongDoubleArraySizeInBytes(kvs)];
            MessageUtils.writeLongDoubleArray(kvs, bytes, 0);
            accumulatedKvsState.add(bytes);
        }
    }

    private void processPushRequest(byte[] pushRpc) {
        MessageType type = MessageUtils.getMessageType(pushRpc);
        if (type == MessageType.ZEROS_TO_PUSH) {
            ZerosToPushM zerosToPush = ZerosToPushM.fromBytes(pushRpc);
            Preconditions.checkState(serverId == zerosToPush.serverId);

            long start = zerosToPush.startIndex;
            long end = zerosToPush.endIndex;
            if (zerosToPush.workerId == 0) {
                modelUpdater.open(start, end);
            }
        } else if (type == MessageType.KVS_TO_PUSH) {
            futuresInEpoch.add(fixedThreadPool.submit(() -> processPushedKvs(pushRpc)));
        } else {
            throw new UnsupportedOperationException("Unsupported message type: " + type + ".");
        }
    }

    private Object processPushedKvs(byte[] pushKv) {
        KVsToPushM kvsToPush = KVsToPushM.fromBytes(pushKv);
        Preconditions.checkState(kvsToPush.serverId == serverId);
        long threadId = Thread.currentThread().getId();
        accumulatedKvsByThreadId.putIfAbsent(threadId, new Long2DoubleOpenHashMap());
        Long2DoubleOpenHashMap tmpGrad = accumulatedKvsByThreadId.get(threadId);

        Tuple2<long[], double[]> pushedGrad = kvsToPush.kvs;
        long[] indices = pushedGrad.f0;
        double[] values = pushedGrad.f1;
        for (int i = 0; i < indices.length; i++) {
            tmpGrad.merge(indices[i], values[i], Double::sum);
        }

        return new Object();
    }

    private Object processPullRequest(byte[] bytesData) {
        IndicesToPullM sparsePullModeM = IndicesToPullM.fromBytes(bytesData);
        Preconditions.checkState(serverId == sparsePullModeM.serverId);
        int workerId = sparsePullModeM.workerId;
        long[] indices = sparsePullModeM.indicesToPull;
        double[] pulledValues = modelUpdater.handlePull(indices);
        ValuesPulledM pulledModelM = new ValuesPulledM(serverId, workerId, pulledValues);
        StreamRecord<Tuple2<Integer, byte[]>> record =
                new StreamRecord<>(Tuple2.of(workerId, pulledModelM.toBytes()));

        // Holds the lock for output.
        synchronized (lock) {
            output.collect(record);
        }
        return new Object();
    }

    private Tuple2<long[], double[]> mergeKvsFromAllThreads(
            Iterator<Long2DoubleOpenHashMap> kvsFromAllThreads) {
        while (kvsFromAllThreads.hasNext()) {
            Long2DoubleOpenHashMap kv = kvsFromAllThreads.next();
            for (Map.Entry<Long, Double> entry : kv.entrySet()) {
                accumulatedKvs.merge(entry.getKey(), entry.getValue(), Double::sum);
            }
            kv.clear();
        }
        long[] indices = new long[accumulatedKvs.size()];
        double[] values = new double[indices.length];
        int idx = 0;
        for (Map.Entry<Long, Double> entry : accumulatedKvs.entrySet()) {
            indices[idx] = entry.getKey();
            values[idx] = entry.getValue();
            idx++;
        }
        return Tuple2.of(indices, values);
    }
}
