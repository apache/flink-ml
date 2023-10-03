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
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.iterations.AllReduceStage;
import org.apache.flink.ml.common.ps.iterations.IterationStage;
import org.apache.flink.ml.common.ps.iterations.IterationStageList;
import org.apache.flink.ml.common.ps.iterations.MLSession;
import org.apache.flink.ml.common.ps.iterations.ProcessStage;
import org.apache.flink.ml.common.ps.iterations.PullStage;
import org.apache.flink.ml.common.ps.iterations.PushStage;
import org.apache.flink.ml.common.ps.iterations.ReduceScatterStage;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.ml.common.ps.utils.ProxySideOutput;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.util.ResettableIterator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import java.util.Iterator;
import java.util.function.Function;

/**
 * The worker operator that executes the iterative machine learning process following {@link
 * IterationStageList}.
 *
 * <p>In detail, the worker operator is responsible for the following:
 *
 * <ul>
 *   <li>Caches the training data.
 *   <li>Initializes the {@link MLSession}.
 *   <li>Splits the {@link IterationStageList} by {@link PullStage}, {@link AllReduceStage} and
 *       {@link ReduceScatterStage} into multiple sequences and map it into flink-ml-iterations.
 *   <li>Executes the process function in each {@link ProcessStage}.
 *   <li>Executes the push/pull/all-reduce/reduce-scatter request in {@link PushStage}, {@link
 *       PullStage}, {@link AllReduceStage} and {@link ReduceScatterStage}. which talks to servers,
 *       by reading/writing {@link MLSession}.
 * </ul>
 */
public class WorkerOperator<DT, SessionT extends MLSession> extends AbstractStreamOperator<byte[]>
        implements TwoInputStreamOperator<DT, byte[], byte[]>, IterationListener<byte[]> {
    /** The user defined iteration logic. */
    private final IterationStageList<SessionT> iterationStages;
    /**
     * Iteration id in terms of {@link IterationStageList}. When we finished processing all stages
     * in stageList, the iteration id increments by one.
     */
    private int iterationId;

    /** The id of the stages to execute in iterationStages. */
    private int nextStageToExecute = 0;

    private ListState<Integer> nextStageToExecuteState;

    /** The agent for each worker to talk with servers. */
    private transient ServerAgent serverAgent;
    /** Number of servers that this worker needs to talk to. */
    private final int numServers;
    /** The hash function to distribute keys to servers. */
    private transient Function<Long, Integer> hashFunc;

    /** The cached training data. */
    private ListStateWithCache<DT> trainDataState;

    /**
     * Number of segments received from servers for the current request. For each request, a worker
     * should receive one segment from each server.
     */
    private int numSegmentsReceived = 0;

    private ListState<Integer> numSegmentsReceivedState;

    /**
     * The memory store for pull answer. For a pull request, each received segment will be filled to
     * the user provided buffer.
     */
    private double[] pulledResult;

    private ListState<double[]> pulledResultState;

    /** The state store for the all-reduce/reduce-scatter results. */
    private ListState<byte[]> reducedResult;

    public WorkerOperator(IterationStageList<SessionT> iterationStages, int numServers) {
        this.iterationStages = iterationStages;
        this.numServers = numServers;
    }

    @Override
    public void open() {
        int workerId = getRuntimeContext().getIndexOfThisSubtask();
        int numWorkers = getRuntimeContext().getNumberOfParallelSubtasks();
        this.hashFunc = key -> (int) (Math.abs(key % numServers));
        this.serverAgent = new ServerAgent(workerId, numServers, hashFunc, output);
        iterationStages.session.setWorldInfo(workerId, numWorkers);
        iterationStages.session.setOutput(new ProxySideOutput(output));
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<byte[]> collector) throws Exception {
        if (epochWatermark == 0) {
            iterationStages.session.setInputData(new ResettableTrainDataIterator<>(trainDataState));
            nextStageToExecute = processIterationStages(nextStageToExecute, iterationStages);
        }
    }

    @Override
    public void onIterationTerminated(Context context, Collector<byte[]> collector) {
        trainDataState.clear();
    }

    @Override
    public void processElement1(StreamRecord<DT> streamRecord) throws Exception {
        trainDataState.add(streamRecord.getValue());
    }

    @Override
    public void processElement2(StreamRecord<byte[]> streamRecord) throws Exception {
        Message message = new Message(streamRecord.getValue());
        IterationStage stage =
                iterationStages.stageList.get(
                        nextStageToExecute % iterationStages.stageList.size());

        boolean proceedToNextStage;
        if (stage instanceof PullStage) {
            proceedToNextStage = onPullResponse(message, (PullStage) stage);
        } else if (stage instanceof AllReduceStage) {
            proceedToNextStage = onAllReduceResponse(message, (AllReduceStage<?>) stage);
        } else if (stage instanceof ReduceScatterStage) {
            proceedToNextStage = onReduceScatterResponse(message, (ReduceScatterStage<?>) stage);
        } else {
            throw new IllegalStateException(
                    "Illegal stage type: %s" + stage.getClass().getSimpleName() + ".");
        }

        if (proceedToNextStage) {
            nextStageToExecute++;
            nextStageToExecute = processIterationStages(nextStageToExecute, iterationStages);
        }
    }

    private boolean onPullResponse(Message message, PullStage pullStage) {
        numSegmentsReceived++;
        double[] segment = message.getValuesInDoubleArray();
        if (segment.length != 0) {
            if (pullStage.aggregator != null) {
                if (pulledResult.length == 0) {
                    pulledResult = segment;
                } else {
                    pulledResult = pullStage.aggregator.merge(segment, pulledResult);
                }
            } else {
                SharedLongArray keys = pullStage.keys.get();
                SharedDoubleArray values = pullStage.values.get();
                int serverId = message.getServerId();
                long[] keysArray = keys.elements();

                if (pulledResult.length == 0) {
                    pulledResult = values.elements();
                }

                int numDoublesPerKey = values.size() / keys.size();
                // Copy the response from one server to the result array.
                int idxInLocalPull = 0;
                for (int i = 0; i < keys.size(); i++) {
                    if (hashFunc.apply(keysArray[i]) == serverId) {
                        System.arraycopy(
                                segment,
                                idxInLocalPull * numDoublesPerKey,
                                pulledResult,
                                i * numDoublesPerKey,
                                numDoublesPerKey);
                        idxInLocalPull++;
                    }
                }
            }
        }

        if (numSegmentsReceived == numServers) {
            SharedDoubleArray pullPlaceHolder = pullStage.values.get();
            System.arraycopy(
                    pulledResult, 0, pullPlaceHolder.elements(), 0, pullPlaceHolder.size());

            pulledResult = new double[0];
            numSegmentsReceived = 0;
            return true;
        }
        return false;
    }

    private <V> boolean onAllReduceResponse(Message message, AllReduceStage<V> allReduceStage)
            throws Exception {
        numSegmentsReceived++;
        reducedResult.add(message.bytes);

        if (numSegmentsReceived == numServers) {
            Message assembled = Message.assembleMessages(reducedResult.get().iterator());
            V[] reduceResult = assembled.getValues(allReduceStage.typeSerializer);
            System.arraycopy(reduceResult, 0, allReduceStage.recvBuf.get(), 0, reduceResult.length);
            reducedResult.clear();
            numSegmentsReceived = 0;
            return true;
        }
        return false;
    }

    private <V> boolean onReduceScatterResponse(
            Message message, ReduceScatterStage<V> reduceScatterStage) throws Exception {
        numSegmentsReceived++;
        reducedResult.add(message.bytes);

        if (numSegmentsReceived == numServers) {
            Message assembled = Message.assembleMessages(reducedResult.get().iterator());
            V[] reduceResult = assembled.getValues(reduceScatterStage.typeSerializer);
            System.arraycopy(
                    reduceResult, 0, reduceScatterStage.recvBuf.get(), 0, reduceResult.length);
            reducedResult.clear();
            numSegmentsReceived = 0;
            return true;
        }
        return false;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        trainDataState =
                new ListStateWithCache<>(
                        (getOperatorConfig().getTypeSerializerIn(0, getClass().getClassLoader())),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        config.getOperatorID());

        numSegmentsReceivedState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("numSegmentsReceivedState", Types.INT));
        numSegmentsReceived =
                OperatorStateUtils.getUniqueElement(
                                numSegmentsReceivedState, "numSegmentsReceivedState")
                        .orElse(0);

        nextStageToExecuteState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("nextStageToExecuteState", Types.INT));

        nextStageToExecute =
                OperatorStateUtils.getUniqueElement(
                                nextStageToExecuteState, "nextStageToExecuteState")
                        .orElse(0);

        iterationStages.session.initializeState(context);

        pulledResultState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "pulledResultState",
                                        PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
        pulledResult =
                OperatorStateUtils.getUniqueElement(pulledResultState, "pulledResultState")
                        .orElse(new double[0]);

        reducedResult =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "reducedResult",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);

        numSegmentsReceivedState.clear();
        numSegmentsReceivedState.add(numSegmentsReceived);

        nextStageToExecuteState.clear();
        nextStageToExecuteState.add(nextStageToExecute);

        trainDataState.snapshotState(context);
        iterationStages.session.snapshotState(context);

        pulledResultState.clear();
        pulledResultState.add(pulledResult);
    }

    /**
     * Processes the stages described in the given iterationStages from the given nextStage id. This
     * function processes the stages until it meets a {@link PullStage}, {@link AllReduceStage} or
     * {@link ReduceScatterStage}.
     *
     * @param nextStageToExecute id of the next stage to execute in the given iteration stages.
     * @param iterationStages iteration stages used to describe the training logic.
     * @return the id of the next stage to execute.
     */
    @SuppressWarnings("unchecked")
    private <V> int processIterationStages(
            int nextStageToExecute, IterationStageList<SessionT> iterationStages) throws Exception {
        while (true) {
            if (nextStageToExecute > 0
                    && nextStageToExecute % iterationStages.stageList.size() == 0) {
                iterationId = nextStageToExecute / iterationStages.stageList.size();
                iterationStages.session.setIterationId(iterationId);
                if (iterationStages.shouldTerminate.apply(iterationStages.session)) {
                    return -1;
                }
            }
            IterationStage stage =
                    iterationStages.stageList.get(
                            nextStageToExecute % iterationStages.stageList.size());

            // We are not incrementing nextStageToExecute for
            // PullStage/AllReduceStage/ReduceScatterStage, since we
            // need to wait for response from servers.
            if (stage instanceof PullStage) {
                PullStage pullStage = ((PullStage) stage);
                serverAgent.pull(pullStage.keys.get(), nextStageToExecute);
                return nextStageToExecute;

            } else if (stage instanceof AllReduceStage) {
                AllReduceStage<V> allReduceStage = (AllReduceStage<V>) stage;
                if (iterationId % allReduceStage.executionInterval == 0) {
                    serverAgent.reduce(
                            allReduceStage.sendBuf.get(),
                            allReduceStage.typeSerializer,
                            nextStageToExecute);
                    return nextStageToExecute;
                } else {
                    nextStageToExecute++;
                }

            } else if (stage instanceof ReduceScatterStage) {
                ReduceScatterStage<V> reduceScatterStage = (ReduceScatterStage<V>) stage;
                if (iterationId % reduceScatterStage.executionInterval == 0) {
                    serverAgent.reduce(
                            reduceScatterStage.sendBuf.get(),
                            reduceScatterStage.typeSerializer,
                            nextStageToExecute);
                    return nextStageToExecute;
                } else {
                    nextStageToExecute++;
                }
            } else if (stage instanceof PushStage) {
                PushStage pushStage = (PushStage) stage;
                serverAgent.push(pushStage.keys.get(), pushStage.values.get(), nextStageToExecute);
                nextStageToExecute++;
            } else if (stage instanceof ProcessStage) {
                ((ProcessStage<SessionT>) stage).process(iterationStages.session);
                nextStageToExecute++;
            } else {
                throw new IllegalStateException(
                        "Illegal type of IterationStage: + "
                                + stage.getClass().getSimpleName()
                                + ".");
            }
        }
    }

    /** A resettable iterator for {@link ListStateWithCache}. */
    private static class ResettableTrainDataIterator<T> implements ResettableIterator<T> {
        private final ListStateWithCache<T> data;
        private Iterator<T> dataIterator;

        public ResettableTrainDataIterator(ListStateWithCache<T> data) throws Exception {
            this.data = data;
            this.dataIterator = data.get().iterator();
        }

        @Override
        public void reset() {
            try {
                this.dataIterator = data.get().iterator();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public boolean hasNext() {
            return dataIterator.hasNext();
        }

        @Override
        public T next() {
            return dataIterator.next();
        }
    }
}
