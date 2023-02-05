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

package org.apache.flink.iteration.progresstrack;

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.ParallelSourceFunction;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractInput;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorV2;
import org.apache.flink.streaming.api.operators.ChainingStrategy;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.transformations.MultipleInputTransformation;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.TestLogger;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Test the {@link OperatorEpochWatermarkTracker} is created correctly according to the topology.
 */
public class OperatorEpochWatermarkTrackerFactoryTest extends TestLogger {

    private static OperatorEpochWatermarkTracker lastProgressTracker;

    private StreamExecutionEnvironment env;

    @Before
    public void before() {
        env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.getConfig().disableGenericTypes();
    }

    @Test
    public void testChainedOperator() throws Exception {
        env.setParallelism(1);
        env.addSource(new EmptySource())
                .transform(
                        "tracking",
                        BasicTypeInfo.INT_TYPE_INFO,
                        new OneInputProgressTrackingOperator(ChainingStrategy.ALWAYS));
        env.execute();

        checkNumberOfInput(new int[] {1});
    }

    @Test
    public void testOneInputOperator() throws Exception {
        env.addSource(new EmptySource())
                .setParallelism(4)
                .transform(
                        "tracking",
                        BasicTypeInfo.INT_TYPE_INFO,
                        new OneInputProgressTrackingOperator(ChainingStrategy.NEVER))
                .setParallelism(1);
        env.execute();

        checkNumberOfInput(new int[] {4});
    }

    @Test
    public void testUnionedOneInput() throws Exception {
        env.addSource(new EmptySource())
                .setParallelism(4)
                .union(env.addSource(new EmptySource()).setParallelism(3))
                .union(env.addSource(new EmptySource()).setParallelism(2))
                .transform(
                        "tracking",
                        BasicTypeInfo.INT_TYPE_INFO,
                        new OneInputProgressTrackingOperator(ChainingStrategy.NEVER))
                .setParallelism(1);
        env.execute();

        checkNumberOfInput(new int[] {9});
    }

    @Test
    public void testTwoInputOperator() throws Exception {
        env.addSource(new EmptySource())
                .setParallelism(4)
                .connect(env.addSource(new EmptySource()).setParallelism(3))
                .transform(
                        "tracking",
                        BasicTypeInfo.INT_TYPE_INFO,
                        new TwoInputProgressTrackingOperator())
                .setParallelism(1);
        env.execute();

        checkNumberOfInput(new int[] {4, 3});
    }

    @Test
    public void testUnionedTwoInputOperator() throws Exception {
        env.addSource(new EmptySource())
                .setParallelism(4)
                .union(env.addSource(new EmptySource()).setParallelism(2))
                .connect(env.addSource(new EmptySource()).setParallelism(3))
                .transform(
                        "tracking",
                        BasicTypeInfo.INT_TYPE_INFO,
                        new TwoInputProgressTrackingOperator())
                .setParallelism(1);
        env.execute();

        checkNumberOfInput(new int[] {6, 3});
    }

    @Test
    public void testMultipleInputOperator() throws Exception {
        DataStream<Integer> first =
                env.addSource(new EmptySource())
                        .setParallelism(4)
                        .union(env.addSource(new EmptySource()).setParallelism(2));
        DataStream<Integer> second =
                env.addSource(new EmptySource())
                        .setParallelism(2)
                        .union(env.addSource(new EmptySource()).setParallelism(3));
        DataStream<Integer> third = env.addSource(new EmptySource()).setParallelism(10);

        MultipleInputTransformation<Integer> multipleInputTransformation =
                new MultipleInputTransformation<>(
                        "tracking",
                        new MultipleInputProgressTrackingOperatorFactory(3),
                        BasicTypeInfo.INT_TYPE_INFO,
                        1);
        multipleInputTransformation.addInput(first.getTransformation());
        multipleInputTransformation.addInput(second.getTransformation());
        multipleInputTransformation.addInput(third.getTransformation());
        env.addOperator(multipleInputTransformation);
        env.execute();

        checkNumberOfInput(new int[] {6, 5, 10});
    }

    private void checkNumberOfInput(int[] numberOfInputs) {
        assertNotNull(lastProgressTracker);
        assertArrayEquals(numberOfInputs, lastProgressTracker.getNumberOfInputs());
    }

    private static class EmptySource implements ParallelSourceFunction<Integer> {

        @Override
        public void run(SourceContext<Integer> ctx) throws Exception {}

        @Override
        public void cancel() {}
    }

    private static class OneInputProgressTrackingOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Integer, Integer> {

        public OneInputProgressTrackingOperator(ChainingStrategy chainingStrategy) {
            this.chainingStrategy = chainingStrategy;
        }

        @Override
        public void setup(
                StreamTask<?, ?> containingTask,
                StreamConfig config,
                Output<StreamRecord<Integer>> output) {
            super.setup(containingTask, config, output);
            lastProgressTracker =
                    OperatorEpochWatermarkTrackerFactory.create(
                            config, containingTask, (ignored) -> {});
        }

        @Override
        public void processElement(StreamRecord<Integer> element) throws Exception {}
    }

    private static class TwoInputProgressTrackingOperator extends AbstractStreamOperator<Integer>
            implements TwoInputStreamOperator<Integer, Integer, Integer> {

        @Override
        public void setup(
                StreamTask<?, ?> containingTask,
                StreamConfig config,
                Output<StreamRecord<Integer>> output) {
            super.setup(containingTask, config, output);
            lastProgressTracker =
                    OperatorEpochWatermarkTrackerFactory.create(
                            config, containingTask, (ignored) -> {});
        }

        @Override
        public void processElement1(StreamRecord<Integer> element) throws Exception {}

        @Override
        public void processElement2(StreamRecord<Integer> element) throws Exception {}
    }

    private static class MultipleInputProgressTrackingOperator
            extends AbstractStreamOperatorV2<Integer>
            implements MultipleInputStreamOperator<Integer> {

        private final int numberOfInputs;

        public MultipleInputProgressTrackingOperator(
                StreamOperatorParameters<Integer> parameters, int numberOfInputs) {
            super(parameters, numberOfInputs);
            this.numberOfInputs = numberOfInputs;
            lastProgressTracker =
                    OperatorEpochWatermarkTrackerFactory.create(
                            config, parameters.getContainingTask(), (ignored) -> {});
        }

        @Override
        public List<Input> getInputs() {
            List<Input> inputs = new ArrayList<>();
            for (int i = 0; i < numberOfInputs; ++i) {
                inputs.add(
                        new AbstractInput(this, i + 1) {
                            @Override
                            public void processElement(StreamRecord element) throws Exception {}
                        });
            }
            return inputs;
        }
    }

    private static class MultipleInputProgressTrackingOperatorFactory
            extends AbstractStreamOperatorFactory<Integer> {

        private final int numberOfInputs;

        public MultipleInputProgressTrackingOperatorFactory(int numberOfInputs) {
            this.numberOfInputs = numberOfInputs;
        }

        @Override
        public <T extends StreamOperator<Integer>> T createStreamOperator(
                StreamOperatorParameters<Integer> parameters) {
            return (T) new MultipleInputProgressTrackingOperator(parameters, numberOfInputs);
        }

        @Override
        public Class<? extends StreamOperator> getStreamOperatorClass(ClassLoader classLoader) {
            return MultipleInputProgressTrackingOperator.class;
        }
    }
}
