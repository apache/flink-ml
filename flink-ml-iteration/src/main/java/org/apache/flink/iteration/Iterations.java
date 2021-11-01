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

package org.apache.flink.iteration;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.iteration.operator.HeadOperatorFactory;
import org.apache.flink.iteration.operator.InputOperator;
import org.apache.flink.iteration.operator.OperatorWrapper;
import org.apache.flink.iteration.operator.OutputOperator;
import org.apache.flink.iteration.operator.TailOperator;
import org.apache.flink.iteration.operator.allround.AllRoundOperatorWrapper;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.streaming.api.transformations.OneInputTransformation;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.apache.flink.util.Preconditions.checkState;

/**
 * A helper class to create iterations. To construct an iteration, Users are required to provide
 *
 * <ul>
 *   <li>initVariableStreams: the initial values of the variable data streams which would be updated
 *       in each round.
 *   <li>dataStreams: the other data streams used inside the iteration, but would not be updated.
 *   <li>iterationBody: specifies the subgraph to update the variable streams and the outputs.
 * </ul>
 *
 * <p>The iteration body will be invoked with two parameters: The first parameter is a list of input
 * variable streams, which are created as the union of the initial variable streams and the
 * corresponding feedback variable streams (returned by the iteration body); The second parameter is
 * the data streams given to this method.
 *
 * <p>During the execution of iteration body, each of the records involved in the iteration has an
 * epoch attached, which is mark the progress of the iteration. The epoch is computed as:
 *
 * <ul>
 *   <li>All records in the initial variable streams and initial data streams has epoch = 0.
 *   <li>For any record emitted by this operator into a non-feedback stream, the epoch of this
 *       emitted record = the epoch of the input record that triggers this emission. If this record
 *       is emitted by onEpochWatermarkIncremented(), then the epoch of this record =
 *       epochWatermark.
 *   <li>For any record emitted by this operator into a feedback variable stream, the epoch of the
 *       emitted record = the epoch of the input record that triggers this emission + 1.
 * </ul>
 *
 * <p>The framework would given the notification at the end of each epoch for operators and UDFs
 * that implements {@link IterationListener}.
 *
 * <p>The limitation of constructing the subgraph inside the iteration body could be refer in {@link
 * IterationBody}.
 *
 * <p>An example of the iteration is like:
 *
 * <pre>{@code
 * DataStreamList result = Iterations.iterateUnboundedStreams(
 *  DataStreamList.of(first, second),
 *  DataStreamList.of(third),
 *  (variableStreams, dataStreams) -> {
 *      ...
 *      return new IterationBodyResult(
 *          DataStreamList.of(firstFeedback, secondFeedback),
 *          DataStreamList.of(output));
 *  }
 *  result.<Integer>get(0).addSink(...);
 * }</pre>
 */
@Experimental
public class Iterations {

    /**
     * This method uses an iteration body to process records in possibly unbounded data streams. The
     * iteration would not terminate if at least one of its inputs is unbounded. Otherwise it will
     * terminated after all the inputs are terminated and no more records are iterating.
     *
     * @param initVariableStreams The initial variable streams, which is merged with the feedback
     *     variable streams before being used as the 1st parameter to invoke the iteration body.
     * @param dataStreams The non-variable streams also refer in the {@code body}.
     * @param body The computation logic which takes variable/data streams and returns
     *     feedback/output streams.
     * @return The list of output streams returned by the iteration boy.
     */
    public static DataStreamList iterateUnboundedStreams(
            DataStreamList initVariableStreams, DataStreamList dataStreams, IterationBody body) {
        return createIteration(
                initVariableStreams, dataStreams, body, new AllRoundOperatorWrapper(), false);
    }

    /**
     * This method uses an iteration body to process records in some bounded data streams
     * iteratively until no more records are iterating or the given terminating criteria stream is
     * empty in one round.
     *
     * @param initVariableStreams The initial variable streams, which is merged with the feedback
     *     variable streams before being used as the 1st parameter to invoke the iteration body.
     * @param dataStreams The non-variable streams also refer in the {@code body} and if each of
     *     them needs replayed for each round.
     * @param config The config for the iteration, like whether to re-create the operator on each
     *     round.
     * @param body The computation logic which takes variable/data streams and returns
     *     feedback/output streams.
     * @return The list of output streams returned by the iteration boy.
     */
    public static DataStreamList iterateBoundedStreamsUntilTermination(
            DataStreamList initVariableStreams,
            ReplayableDataStreamList dataStreams,
            IterationConfig config,
            IterationBody body) {
        Preconditions.checkArgument(
                config.getOperatorLifeCycle() == IterationConfig.OperatorLifeCycle.ALL_ROUND);
        Preconditions.checkArgument(dataStreams.getReplayedDataStreams().size() == 0);

        return createIteration(
                initVariableStreams,
                new DataStreamList(dataStreams.getNonReplayedStreams()),
                body,
                new AllRoundOperatorWrapper(),
                true);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static DataStreamList createIteration(
            DataStreamList initVariableStreams,
            DataStreamList dataStreams,
            IterationBody body,
            OperatorWrapper<?, IterationRecord<?>> initialOperatorWrapper,
            boolean mayHaveCriteria) {
        checkState(initVariableStreams.size() > 0, "There should be at least one variable stream");

        IterationID iterationId = new IterationID();

        List<TypeInformation<?>> initVariableTypeInfos = getTypeInfos(initVariableStreams);
        List<TypeInformation<?>> dataStreamTypeInfos = getTypeInfos(dataStreams);

        // Add heads and inputs
        int totalInitVariableParallelism =
                map(
                                initVariableStreams,
                                dataStream ->
                                        dataStream.getParallelism() > 0
                                                ? dataStream.getParallelism()
                                                : dataStream
                                                        .getExecutionEnvironment()
                                                        .getConfig()
                                                        .getParallelism())
                        .stream()
                        .mapToInt(i -> i)
                        .sum();
        DataStreamList initVariableInputs = addInputs(initVariableStreams, false);
        DataStreamList headStreams =
                addHeads(
                        initVariableStreams,
                        initVariableInputs,
                        iterationId,
                        totalInitVariableParallelism,
                        false,
                        0);

        DataStreamList dataStreamInputs = addInputs(dataStreams, true);

        // Create the iteration body. We map the inputs of iteration body into the draft sources,
        // which serve as the start points to build the draft subgraph.
        StreamExecutionEnvironment env = initVariableStreams.get(0).getExecutionEnvironment();
        DraftExecutionEnvironment draftEnv =
                new DraftExecutionEnvironment(env, initialOperatorWrapper);
        DataStreamList draftHeadStreams =
                addDraftSources(headStreams, draftEnv, initVariableTypeInfos);
        DataStreamList draftDataStreamInputs =
                addDraftSources(dataStreamInputs, draftEnv, dataStreamTypeInfos);

        IterationBodyResult iterationBodyResult =
                body.process(draftHeadStreams, draftDataStreamInputs);
        ensuresTransformationAdded(iterationBodyResult.getFeedbackVariableStreams(), draftEnv);
        ensuresTransformationAdded(iterationBodyResult.getOutputStreams(), draftEnv);
        draftEnv.copyToActualEnvironment();

        // Add tails and co-locate them with the heads.
        DataStreamList feedbackStreams =
                getActualDataStreams(iterationBodyResult.getFeedbackVariableStreams(), draftEnv);
        checkState(
                feedbackStreams.size() == initVariableStreams.size(),
                "The number of feedback streams "
                        + feedbackStreams.size()
                        + " does not match the initialized one "
                        + initVariableStreams.size());
        DataStreamList tails = addTails(feedbackStreams, iterationId, 0);
        for (int i = 0; i < headStreams.size(); ++i) {
            String coLocationGroupKey = "co-" + iterationId.toHexString() + "-" + i;
            headStreams.get(i).getTransformation().setCoLocationGroupKey(coLocationGroupKey);
            tails.get(i).getTransformation().setCoLocationGroupKey(coLocationGroupKey);
        }

        checkState(
                mayHaveCriteria || iterationBodyResult.getTerminationCriteria() == null,
                "The current iteration type does not support the termination criteria.");

        if (iterationBodyResult.getTerminationCriteria() != null) {
            addCriteriaStream(
                    iterationBodyResult.getTerminationCriteria(),
                    iterationId,
                    env,
                    draftEnv,
                    initVariableStreams,
                    headStreams,
                    totalInitVariableParallelism);
        }

        return addOutputs(getActualDataStreams(iterationBodyResult.getOutputStreams(), draftEnv));
    }

    private static void addCriteriaStream(
            DataStream<?> draftCriteriaStream,
            IterationID iterationId,
            StreamExecutionEnvironment env,
            DraftExecutionEnvironment draftEnv,
            DataStreamList initVariableStreams,
            DataStreamList headStreams,
            int totalInitVariableParallelism) {
        // deal with the criteria streams
        DataStream<?> terminationCriteria = draftEnv.getActualStream(draftCriteriaStream.getId());
        // It should always has the IterationRecordTypeInfo
        checkState(
                terminationCriteria.getType().getClass().equals(IterationRecordTypeInfo.class),
                "The termination criteria should always return IterationRecord.");
        TypeInformation<?> innerType =
                ((IterationRecordTypeInfo<?>) terminationCriteria.getType()).getInnerTypeInfo();

        DataStream<?> emptyCriteriaSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource())
                        .returns(innerType)
                        .name(terminationCriteria.getTransformation().getName())
                        .setParallelism(terminationCriteria.getParallelism());
        DataStreamList criteriaSources = DataStreamList.of(emptyCriteriaSource);
        DataStreamList criteriaInputs = addInputs(criteriaSources, false);
        DataStreamList criteriaHeaders =
                addHeads(
                        criteriaSources,
                        criteriaInputs,
                        iterationId,
                        totalInitVariableParallelism,
                        true,
                        initVariableStreams.size());

        // Merges the head and the actual criteria stream. This is required since if we have
        // no edges from the criteria head to the criteria tail, the tail might directly received
        // the MAX_EPOCH_WATERMARK without the synchronization of the head.
        DataStream<?> mergedHeadAndCriteria =
                mergeCriteriaHeadAndCriteriaStream(
                        env, criteriaHeaders.get(0), terminationCriteria, innerType);
        DataStreamList criteriaTails =
                addTails(
                        DataStreamList.of(mergedHeadAndCriteria),
                        iterationId,
                        initVariableStreams.size());

        String coLocationGroupKey = "co-" + iterationId.toHexString() + "-cri";
        criteriaHeaders.get(0).getTransformation().setCoLocationGroupKey(coLocationGroupKey);
        criteriaTails.get(0).getTransformation().setCoLocationGroupKey(coLocationGroupKey);

        // Now we notify all the head operators to count the criteria stream.
        setCriteriaParallelism(headStreams, terminationCriteria.getParallelism());
        setCriteriaParallelism(criteriaHeaders, terminationCriteria.getParallelism());
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static DataStream<?> mergeCriteriaHeadAndCriteriaStream(
            StreamExecutionEnvironment env,
            DataStream<?> head,
            DataStream<?> criteriaStream,
            TypeInformation<?> criteriaStreamType) {
        DraftExecutionEnvironment criteriaDraftEnv =
                new DraftExecutionEnvironment(env, new AllRoundOperatorWrapper<>());
        DataStream draftHeadStream = criteriaDraftEnv.addDraftSource(head, criteriaStreamType);
        DataStream draftTerminationCriteria =
                criteriaDraftEnv.addDraftSource(criteriaStream, criteriaStreamType);
        DataStream draftMergedStream =
                draftHeadStream
                        .connect(draftTerminationCriteria)
                        .process(new CriteriaMergeProcessor())
                        .returns(criteriaStreamType)
                        .setParallelism(
                                criteriaStream.getParallelism() > 0
                                        ? criteriaStream.getParallelism()
                                        : env.getConfig().getParallelism())
                        .name("criteria-merge");
        criteriaDraftEnv.copyToActualEnvironment();
        return criteriaDraftEnv.getActualStream(draftMergedStream.getId());
    }

    private static List<TypeInformation<?>> getTypeInfos(DataStreamList dataStreams) {
        return map(dataStreams, DataStream::getType);
    }

    private static DataStreamList addInputs(
            DataStreamList dataStreams, boolean insertMaxEpochWatermark) {
        return new DataStreamList(
                map(
                        dataStreams,
                        dataStream ->
                                dataStream
                                        .transform(
                                                "input-" + dataStream.getTransformation().getName(),
                                                new IterationRecordTypeInfo<>(dataStream.getType()),
                                                new InputOperator(insertMaxEpochWatermark))
                                        .setParallelism(dataStream.getParallelism())));
    }

    private static DataStreamList addHeads(
            DataStreamList variableStreams,
            DataStreamList inputStreams,
            IterationID iterationId,
            int totalInitVariableParallelism,
            boolean isCriteriaStream,
            int startHeaderIndex) {

        return new DataStreamList(
                map(
                        inputStreams,
                        (index, dataStream) ->
                                ((SingleOutputStreamOperator<IterationRecord<?>>) dataStream)
                                        .transform(
                                                "head-"
                                                        + variableStreams
                                                                .get(index)
                                                                .getTransformation()
                                                                .getName(),
                                                (IterationRecordTypeInfo) dataStream.getType(),
                                                new HeadOperatorFactory(
                                                        iterationId,
                                                        startHeaderIndex + index,
                                                        isCriteriaStream,
                                                        totalInitVariableParallelism))
                                        .setParallelism(dataStream.getParallelism())));
    }

    private static DataStreamList addTails(
            DataStreamList dataStreams, IterationID iterationId, int startIndex) {
        return new DataStreamList(
                map(
                        dataStreams,
                        (index, dataStream) ->
                                ((DataStream<IterationRecord<?>>) dataStream)
                                        .transform(
                                                "tail-" + dataStream.getTransformation().getName(),
                                                new IterationRecordTypeInfo(dataStream.getType()),
                                                new TailOperator(iterationId, startIndex + index))
                                        .setParallelism(dataStream.getParallelism())));
    }

    private static DataStreamList addOutputs(DataStreamList dataStreams) {
        return new DataStreamList(
                map(
                        dataStreams,
                        (index, dataStream) -> {
                            IterationRecordTypeInfo<?> inputType =
                                    (IterationRecordTypeInfo<?>) dataStream.getType();
                            return dataStream
                                    .transform(
                                            "output-" + dataStream.getTransformation().getName(),
                                            inputType.getInnerTypeInfo(),
                                            new OutputOperator())
                                    .setParallelism(dataStream.getParallelism());
                        }));
    }

    private static DataStreamList addDraftSources(
            DataStreamList dataStreams,
            DraftExecutionEnvironment draftEnv,
            List<TypeInformation<?>> typeInfos) {

        return new DataStreamList(
                map(
                        dataStreams,
                        (index, dataStream) ->
                                draftEnv.addDraftSource(dataStream, typeInfos.get(index))));
    }

    private static void ensuresTransformationAdded(
            DataStreamList dataStreams, DraftExecutionEnvironment draftEnv) {
        map(
                dataStreams,
                dataStream -> {
                    draftEnv.addOperatorIfNotExists(dataStream.getTransformation());
                    return null;
                });
    }

    private static void setCriteriaParallelism(
            DataStreamList headStreams, int criteriaParallelism) {
        map(
                headStreams,
                dataStream -> {
                    ((HeadOperatorFactory)
                                    ((OneInputTransformation) dataStream.getTransformation())
                                            .getOperatorFactory())
                            .setCriteriaStreamParallelism(criteriaParallelism);
                    return null;
                });
    }

    private static DataStreamList getActualDataStreams(
            DataStreamList draftStreams, DraftExecutionEnvironment draftEnv) {
        return new DataStreamList(
                map(draftStreams, dataStream -> draftEnv.getActualStream(dataStream.getId())));
    }

    private static <R> List<R> map(DataStreamList dataStreams, Function<DataStream<?>, R> mapper) {
        return map(dataStreams, (i, dataStream) -> mapper.apply(dataStream));
    }

    private static <R> List<R> map(
            DataStreamList dataStreams, BiFunction<Integer, DataStream<?>, R> mapper) {
        List<R> results = new ArrayList<>(dataStreams.size());
        for (int i = 0; i < dataStreams.size(); ++i) {
            DataStream<?> dataStream = dataStreams.get(i);
            results.add(mapper.apply(i, dataStream));
        }

        return results;
    }

    private static class CriteriaMergeProcessor extends CoProcessFunction<Integer, Object, Object> {

        @Override
        public void processElement1(Integer value, Context ctx, Collector<Object> out)
                throws Exception {
            // Ignores all the records from the head side-output.
        }

        @Override
        public void processElement2(Object value, Context ctx, Collector<Object> out)
                throws Exception {
            // Bypasses all the records from the actual criteria stream.
            out.collect(value);
        }
    }
}
