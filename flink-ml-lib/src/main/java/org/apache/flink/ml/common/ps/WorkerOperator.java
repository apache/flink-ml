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
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.message.ValuesPulledM;
import org.apache.flink.ml.common.ps.training.IterationStage;
import org.apache.flink.ml.common.ps.training.IterationStageList;
import org.apache.flink.ml.common.ps.training.ProcessStage;
import org.apache.flink.ml.common.ps.training.PullStage;
import org.apache.flink.ml.common.ps.training.PushStage;
import org.apache.flink.ml.common.ps.training.TrainingContext;
import org.apache.flink.ml.util.Bits;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.util.ResettableIterator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.util.Iterator;

/**
 * The worker operator that executes the machine learning training process following {@link
 * IterationStageList}.
 *
 * <p>In detail, the worker operator is responsible for the following:
 *
 * <ul>
 *   <li>Caches the training data.
 *   <li>Initializes the {@link TrainingContext}.
 *   <li>Splits the {@link IterationStageList} by {@link PullStage} into multiple sequences and map
 *       it into flink-ml-iterations.
 *   <li>Executes the process function in each {@link ProcessStage}.
 *   <li>Executes the push/pull request in {@link PushStage} and {@link PullStage} and talk to
 *       servers, by reading/writing {@link TrainingContext}.
 * </ul>
 */
public class WorkerOperator<DT, ContextT extends TrainingContext>
        extends AbstractStreamOperator<Tuple2<Integer, byte[]>>
        implements TwoInputStreamOperator<DT, byte[], Tuple2<Integer, byte[]>>,
                IterationListener<Tuple2<Integer, byte[]>> {
    /** Number of servers that this worker needs to talk to. */
    private final int numServers;

    /** The agent for each worker to talk with servers. */
    private ServerAgent serverAgent;

    /** The user defined iteration logic. */
    private final IterationStageList<ContextT> iterationStages;

    /**
     * Iteration id in terms of {@link IterationStageList}. When we finished processing all stages
     * in stageList, the iteration id increments by one.
     */
    private int iterationId;

    private ListState<Integer> iterationIdState;

    /** The id of the stages to execute in iterationStages. */
    private int nextStageToExecute = 0;

    private ListState<Integer> nextStageToExecuteState;

    /** The cached training data. */
    private ListStateWithCache<DT> trainDataState;

    /** The feedback array from iterations. */
    private byte[] feedback;

    private ListState<byte[]> feedbackState;

    /** Dimension of the model. */
    private long modelDim = 0;

    private ListState<Long> modelDimState;

    public WorkerOperator(IterationStageList<ContextT> iterationStages, int numServers) {
        this.iterationStages = iterationStages;
        this.numServers = numServers;
    }

    @Override
    public void open() {
        int numTasks = getRuntimeContext().getNumberOfParallelSubtasks();
        int workerId = getRuntimeContext().getIndexOfThisSubtask();
        this.serverAgent = new ServerAgent(workerId, output);
        iterationStages.context.setWorldInfo(workerId, numTasks);
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Tuple2<Integer, byte[]>> collector)
            throws Exception {
        if (epochWatermark == 0) {
            modelDim = Bits.getLong(feedback, 0);
            serverAgent.setPartitioner(new RangePartitioner(modelDim, numServers));
            serverAgent.zeros();
            iterationStages.context.setInputData(new ResettableTrainDataIterator<>(trainDataState));
            nextStageToExecute = processTrainingStage(nextStageToExecute, iterationStages);
        }
    }

    @Override
    public void onIterationTerminated(
            Context context, Collector<Tuple2<Integer, byte[]>> collector) {
        trainDataState.clear();
    }

    @Override
    public void processElement1(StreamRecord<DT> streamRecord) throws Exception {
        trainDataState.add(streamRecord.getValue());
    }

    @Override
    public void processElement2(StreamRecord<byte[]> streamRecord) throws Exception {
        feedback = streamRecord.getValue();
        if (modelDim > 0) {
            // Decodes the pulled method and put it in training context.
            PullStage pullStage = (PullStage) iterationStages.stageList.get(nextStageToExecute);
            ValuesPulledM valuesPulledMessage = ValuesPulledM.fromBytes(streamRecord.getValue());
            Preconditions.checkState(
                    getRuntimeContext().getIndexOfThisSubtask() == valuesPulledMessage.workerId);
            pullStage.valuesConsumer.accept(valuesPulledMessage.valuesPulled);
            nextStageToExecute++;

            nextStageToExecute = processTrainingStage(nextStageToExecute, iterationStages);
        }
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        feedbackState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "feedbackArrayState",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));
        OperatorStateUtils.getUniqueElement(feedbackState, "feedbackArrayState")
                .ifPresent(x -> feedback = x);

        trainDataState =
                new ListStateWithCache<>(
                        (getOperatorConfig().getTypeSerializerIn(0, getClass().getClassLoader())),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        config.getOperatorID());

        nextStageToExecuteState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("nextStageToExecuteState", Types.INT));
        nextStageToExecute =
                OperatorStateUtils.getUniqueElement(
                                nextStageToExecuteState, "nextStageToExecuteState")
                        .orElse(0);

        modelDimState =
                context.getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("modelDimState", Types.LONG));
        modelDim = OperatorStateUtils.getUniqueElement(modelDimState, "modelDimState").orElse(0L);

        iterationIdState =
                context.getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("iterationIdState", Types.INT));
        iterationId =
                OperatorStateUtils.getUniqueElement(iterationIdState, "iterationIdState").orElse(0);

        if (modelDim > 0) {
            serverAgent.setPartitioner(new RangePartitioner(modelDim, numServers));
        }

        iterationStages.context.initializeState(context);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        feedbackState.clear();
        if (feedback != null) {
            feedbackState.add(feedback);
        }

        nextStageToExecuteState.clear();
        nextStageToExecuteState.add(nextStageToExecute);
        modelDimState.clear();
        modelDimState.add(modelDim);
        iterationIdState.clear();
        iterationIdState.add(iterationId);

        trainDataState.snapshotState(context);
        iterationStages.context.snapshotState(context);
    }

    /**
     * Processes the stages described in the given iterationStages from the given nextStage id. This
     * function processes the stages until it meets an {@link PullStage}.
     *
     * @param nextStageToExecute id of the next stage to execute in the given iteration stages.
     * @param iterationStages iteration stages used to describe the training logic.
     * @return the id of the next stage to execute.
     */
    @SuppressWarnings("unchecked")
    private int processTrainingStage(
            int nextStageToExecute, IterationStageList<ContextT> iterationStages) throws Exception {
        while (true) {
            if (nextStageToExecute >= iterationStages.stageList.size()) {
                iterationId++;
                iterationStages.context.setIterationId(iterationId);
                if (iterationStages.shouldTerminate.apply(iterationStages.context)) {
                    return -1;
                }
                nextStageToExecute -= iterationStages.stageList.size();
            }
            IterationStage stage = iterationStages.stageList.get(nextStageToExecute);

            if (stage instanceof PullStage) {
                // We are not incrementing nextStageToExecute here, since we will need to pull
                // values from servers.
                PullStage pullStage = ((PullStage) stage);
                serverAgent.pull(pullStage.keysSupplier.get());
                return nextStageToExecute;
            } else if (stage instanceof PushStage) {
                PushStage pushStage = (PushStage) stage;
                serverAgent.pushKVs(pushStage.keysSupplier.get(), pushStage.valuesSupplier.get());
                nextStageToExecute++;
            } else if (stage instanceof ProcessStage) {
                ((ProcessStage<ContextT>) stage).process(iterationStages.context);
                nextStageToExecute++;
            } else {
                throw new IllegalStateException(
                        "Illegal type of IterationStage: + " + stage.getClass().getSimpleName());
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
