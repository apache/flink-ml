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

import org.apache.flink.annotation.Internal;
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

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.apache.flink.util.Preconditions.checkState;

/** Creates iteration in a job. */
@Internal
public class IterationFactory {

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static DataStreamList createIteration(
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
