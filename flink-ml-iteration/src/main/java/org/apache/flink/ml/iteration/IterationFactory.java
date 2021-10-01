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

package org.apache.flink.ml.iteration;

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.ml.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.ml.iteration.operator.HeadOperatorFactory;
import org.apache.flink.ml.iteration.operator.InputOperator;
import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.ml.iteration.operator.OutputOperator;
import org.apache.flink.ml.iteration.operator.TailOperator;
import org.apache.flink.ml.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.transformations.OneInputTransformation;
import org.apache.flink.util.OutputTag;

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
                map(initVariableStreams, DataStream::getParallelism).stream()
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

        // Create the iteration body.
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
                "The termination criteria should always returns IterationRecord.");
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
        DataStreamList criteriaTails =
                addTails(
                        DataStreamList.of(terminationCriteria),
                        iterationId,
                        initVariableStreams.size());

        String coLocationGroupKey = "co-" + iterationId.toHexString() + "-cri";
        criteriaHeaders.get(0).getTransformation().setCoLocationGroupKey(coLocationGroupKey);
        criteriaTails.get(0).getTransformation().setCoLocationGroupKey(coLocationGroupKey);

        // Since co-located task must be in the same region, we will have to add a fake op.
        ((SingleOutputStreamOperator<?>) criteriaHeaders.get(0))
                .getSideOutput(new OutputTag<IterationRecord<Integer>>("fake") {})
                .union(
                        ((SingleOutputStreamOperator<?>) criteriaTails.get(0))
                                .getSideOutput(new OutputTag<IterationRecord<Integer>>("fake") {}))
                .map(x -> x)
                .returns(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                .name("criteria-discard")
                .setParallelism(1);

        // Now we notify all the head operators to count the criteria stream.
        setCriteriaParallelism(headStreams, terminationCriteria.getParallelism());
        setCriteriaParallelism(criteriaHeaders, terminationCriteria.getParallelism());
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
        List<R> results = new ArrayList<>();
        for (int i = 0; i < dataStreams.size(); ++i) {
            DataStream<?> dataStream = dataStreams.get(i);
            results.add(mapper.apply(i, dataStream));
        }

        return results;
    }
}
