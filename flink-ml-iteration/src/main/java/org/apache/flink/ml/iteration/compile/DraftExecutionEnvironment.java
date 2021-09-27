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

package org.apache.flink.ml.iteration.compile;

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.dag.Transformation;
import org.apache.flink.core.execution.DefaultExecutorServiceLoader;
import org.apache.flink.ml.iteration.compile.translator.BroadcastStateTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.KeyedBroadcastStateTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.MultipleInputTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.OneInputTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.PartitionTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.ReduceTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.SideOutputTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.TwoInputTransformationTranslator;
import org.apache.flink.ml.iteration.compile.translator.UnionTransformationTranslator;
import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.ml.iteration.utils.ReflectionUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;
import org.apache.flink.streaming.api.graph.StreamGraph;
import org.apache.flink.streaming.api.transformations.BroadcastStateTransformation;
import org.apache.flink.streaming.api.transformations.KeyedBroadcastStateTransformation;
import org.apache.flink.streaming.api.transformations.KeyedMultipleInputTransformation;
import org.apache.flink.streaming.api.transformations.MultipleInputTransformation;
import org.apache.flink.streaming.api.transformations.OneInputTransformation;
import org.apache.flink.streaming.api.transformations.PartitionTransformation;
import org.apache.flink.streaming.api.transformations.ReduceTransformation;
import org.apache.flink.streaming.api.transformations.SideOutputTransformation;
import org.apache.flink.streaming.api.transformations.TwoInputTransformation;
import org.apache.flink.streaming.api.transformations.UnionTransformation;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static org.apache.flink.util.Preconditions.checkState;

/**
 * A specialized stream execution environment that allows users to first construct a subgraph and
 * later copy the transformations into the actual environment. During the copying it could apply
 * some kinds of {@link OperatorWrapper} to change the operators in each transformation.
 */
public class DraftExecutionEnvironment extends StreamExecutionEnvironment {

    @SuppressWarnings("rawtypes")
    private static final Map<Class<? extends Transformation>, DraftTransformationTranslator>
            translators = new HashMap<>();

    static {
        translators.put(
                BroadcastStateTransformation.class, new BroadcastStateTransformationTranslator());
        translators.put(
                KeyedBroadcastStateTransformation.class,
                new KeyedBroadcastStateTransformationTranslator());
        translators.put(
                KeyedMultipleInputTransformation.class,
                new KeyedBroadcastStateTransformationTranslator());
        translators.put(
                MultipleInputTransformation.class, new MultipleInputTransformationTranslator());
        translators.put(OneInputTransformation.class, new OneInputTransformationTranslator());
        translators.put(PartitionTransformation.class, new PartitionTransformationTranslator());
        translators.put(ReduceTransformation.class, new ReduceTransformationTranslator());
        translators.put(SideOutputTransformation.class, new SideOutputTransformationTranslator());
        translators.put(TwoInputTransformation.class, new TwoInputTransformationTranslator());
        translators.put(UnionTransformation.class, new UnionTransformationTranslator());
    }

    private final StreamExecutionEnvironment actualEnv;

    private final Map<Integer, OperatorWrapper<?, ?>> draftWrappers;

    private final Map<Integer, Transformation<?>> draftToActualTransformations;

    private OperatorWrapper<?, ?> currentWrapper;

    public DraftExecutionEnvironment(
            StreamExecutionEnvironment actualEnv, OperatorWrapper<?, ?> initialWrapper) {
        super(
                new DefaultExecutorServiceLoader(),
                ReflectionUtils.getFieldValue(
                        actualEnv, StreamExecutionEnvironment.class, "configuration"),
                ReflectionUtils.getFieldValue(
                        actualEnv, StreamExecutionEnvironment.class, "userClassloader"));
        this.actualEnv = actualEnv;
        this.draftWrappers = new HashMap<>();
        this.draftToActualTransformations = new HashMap<>();

        setParallelism(actualEnv.getParallelism());
        if (actualEnv.getMaxParallelism() > 0) {
            setMaxParallelism(actualEnv.getMaxParallelism());
        }
        setBufferTimeout(actualEnv.getBufferTimeout());

        this.currentWrapper = initialWrapper;
    }

    public OperatorWrapper<?, ?> setCurrentWrapper(OperatorWrapper<?, ?> newWrapper) {
        OperatorWrapper<?, ?> oldWrapper = currentWrapper;
        currentWrapper = newWrapper;
        return oldWrapper;
    }

    @Override
    public void addOperator(Transformation<?> transformation) {
        // Record the wrapper
        recordWrapper(transformation);
        super.addOperator(transformation);
    }

    private void recordWrapper(Transformation<?> transformation) {
        if (draftWrappers.containsKey(transformation.getId())
                || draftToActualTransformations.containsKey(transformation.getId())) {
            return;
        }

        draftWrappers.put(transformation.getId(), currentWrapper);

        for (Transformation<?> input : transformation.getInputs()) {
            recordWrapper(input);
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public <T> DataStream<T> addDraftSource(
            DataStream<?> actualStream, TypeInformation<T> draftOutputType) {
        // Notes that the actual stream are given, thus it does not matter whether
        // the draft sources have the same properties with the actual streams.
        DataStream<T> draftSource =
                addSource(new EmptySource<>())
                        .setParallelism(actualStream.getParallelism())
                        .returns((TypeInformation) draftOutputType);
        addOperator(draftSource.getTransformation());
        draftToActualTransformations.put(draftSource.getId(), actualStream.getTransformation());
        return draftSource;
    }

    public void copyToActualEnvironment() {
        for (Transformation<?> draftTransformation : transformations) {
            transform(draftTransformation);
        }
    }

    public <T> DataStream<T> getActualStream(int draftTransformationId) {
        return new DataStream<T>(actualEnv, getActualTransformation(draftTransformationId));
    }

    @SuppressWarnings("unchecked")
    private <TF extends Transformation<?>> void transform(TF draftTransformation) {
        if (draftToActualTransformations.containsKey(draftTransformation.getId())) {
            return;
        }

        // Ensures the inputs are all transformed
        for (Transformation<?> draftInput : draftTransformation.getInputs()) {
            transform(draftInput);
        }

        OperatorWrapper<?, ?> operatorWrapper =
                Objects.requireNonNull(draftWrappers.get(draftTransformation.getId()));

        DraftTransformationTranslator<TF> transformationTranslator =
                translators.get(draftTransformation.getClass());
        checkState(
                transformationTranslator != null,
                "Unsupported transformation: " + draftTransformation);
        Transformation<?> actualTransformation =
                transformationTranslator.translate(
                        draftTransformation, operatorWrapper, new TranslatorContext());
        actualEnv.addOperator(actualTransformation);
        draftToActualTransformations.put(draftTransformation.getId(), actualTransformation);
    }

    @SuppressWarnings({"unchecked"})
    private <T> Transformation<T> getActualTransformation(int draftTransformationId) {
        return (Transformation<T>)
                Objects.requireNonNull(draftToActualTransformations.get(draftTransformationId));
    }

    @Override
    public JobExecutionResult execute(StreamGraph streamGraph) throws Exception {
        throw new UnsupportedOperationException(
                "Unable to execute with a draft execution environment.");
    }

    private class TranslatorContext implements DraftTransformationTranslator.Context {

        @Override
        public Transformation<?> getActualTransformation(int draftId) {
            return DraftExecutionEnvironment.this.getActualTransformation(draftId);
        }

        @Override
        public ExecutionConfig getExecutionConfig() {
            return DraftExecutionEnvironment.this.getConfig();
        }
    }

    @VisibleForTesting
    static class EmptySource<T> extends RichParallelSourceFunction<T> {

        @Override
        public void run(SourceContext<T> ctx) throws Exception {}

        @Override
        public void cancel() {}
    }
}
