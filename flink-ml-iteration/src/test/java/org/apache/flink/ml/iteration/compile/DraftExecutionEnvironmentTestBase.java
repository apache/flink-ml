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

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.ml.iteration.operator.allround.MultipleInputAllRoundWrapperOperatorTest;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.datastream.MultipleConnectedStreams;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.co.BroadcastProcessFunction;
import org.apache.flink.streaming.api.functions.co.CoMapFunction;
import org.apache.flink.streaming.api.functions.co.KeyedBroadcastProcessFunction;
import org.apache.flink.streaming.api.graph.StreamGraph;
import org.apache.flink.streaming.api.transformations.MultipleInputTransformation;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/** Tests the behavior of the {@link DraftExecutionEnvironment}. */
public abstract class DraftExecutionEnvironmentTestBase {

    protected abstract OperatorWrapper<?, ?> getOperatorWrapper();

    protected abstract void checkWrappedGraph(
            StreamGraph nonWrapped, StreamGraph wrapped, DraftExecutionEnvironment draftEnv);

    @Test
    public void testOneInputTransformation() {
        testWrapper(1, sources -> sources.get(0).map(x -> x).map(x -> x));
    }

    @Test
    public void testKeyedOneInputTransformation() {
        testWrapper(
                1,
                sources ->
                        sources.get(0)
                                .keyBy((KeySelector<Integer, Integer>) integer -> integer)
                                .process(
                                        new KeyedProcessFunction<Integer, Integer, Integer>() {
                                            @Override
                                            public void processElement(
                                                    Integer value,
                                                    Context ctx,
                                                    Collector<Integer> out)
                                                    throws Exception {}
                                        })
                                .map(x -> x));
    }

    @Test
    public void testTwoInputTransformation() {
        testWrapper(
                2,
                sources ->
                        sources.get(0)
                                .connect(sources.get(1))
                                .map(
                                        new CoMapFunction<Integer, Integer, Object>() {
                                            @Override
                                            public Object map1(Integer value) throws Exception {
                                                return null;
                                            }

                                            @Override
                                            public Object map2(Integer value) throws Exception {
                                                return null;
                                            }
                                        })
                                .map(x -> x));
    }

    @Test
    public void testKeyedTwoInputTransformation() {
        testWrapper(
                2,
                sources ->
                        sources.get(0)
                                .keyBy(i -> i)
                                .connect(sources.get(1).keyBy(i -> i))
                                .map(
                                        new CoMapFunction<Integer, Integer, Object>() {
                                            @Override
                                            public Object map1(Integer value) throws Exception {
                                                return null;
                                            }

                                            @Override
                                            public Object map2(Integer value) throws Exception {
                                                return null;
                                            }
                                        })
                                .map(x -> x));
    }

    @Test
    public void testUnionInputStream() {
        testWrapper(
                2,
                sources ->
                        sources.get(0)
                                .keyBy(i -> i)
                                .union(sources.get(1).keyBy(i -> i))
                                .map(x -> x));
    }

    @Test
    public void testBroadcastStateTransformation() {
        testWrapper(
                2,
                sources -> {
                    sources.get(0)
                            .connect(
                                    sources.get(1)
                                            .broadcast(
                                                    new MapStateDescriptor<Integer, Integer>(
                                                            "test", Integer.class, Integer.class)))
                            .process(
                                    new BroadcastProcessFunction<Integer, Integer, Integer>() {
                                        @Override
                                        public void processElement(
                                                Integer value,
                                                ReadOnlyContext ctx,
                                                Collector<Integer> out)
                                                throws Exception {}

                                        @Override
                                        public void processBroadcastElement(
                                                Integer value, Context ctx, Collector<Integer> out)
                                                throws Exception {}
                                    })
                            .map(x -> x);
                });
    }

    @Test
    public void testKeyedBroadcastStateTransformation() {
        testWrapper(
                2,
                sources -> {
                    sources.get(0)
                            .keyBy(i -> i)
                            .connect(
                                    sources.get(1)
                                            .broadcast(
                                                    new MapStateDescriptor<Integer, Integer>(
                                                            "test", Integer.class, Integer.class)))
                            .process(
                                    new KeyedBroadcastProcessFunction<
                                            Integer, Integer, Integer, Integer>() {
                                        @Override
                                        public void processElement(
                                                Integer value,
                                                ReadOnlyContext ctx,
                                                Collector<Integer> out)
                                                throws Exception {}

                                        @Override
                                        public void processBroadcastElement(
                                                Integer value, Context ctx, Collector<Integer> out)
                                                throws Exception {}
                                    })
                            .map(x -> x);
                });
    }

    @Test
    public void testReduceTransformation() {
        testWrapper(
                1,
                sources ->
                        sources.get(0)
                                .keyBy(i -> i)
                                .reduce(
                                        new ReduceFunction<Integer>() {
                                            @Override
                                            public Integer reduce(Integer integer, Integer t1)
                                                    throws Exception {
                                                return null;
                                            }
                                        })
                                .map(x -> x));
    }

    @Test
    public void testSideOutput() {
        testWrapper(
                1,
                sources ->
                        sources.get(0)
                                .process(
                                        new ProcessFunction<Integer, Integer>() {
                                            @Override
                                            public void processElement(
                                                    Integer value,
                                                    Context ctx,
                                                    Collector<Integer> out)
                                                    throws Exception {}
                                        })
                                .getSideOutput(new OutputTag<String>("test") {})
                                .map(x -> x));
    }

    @Test
    public void testMultipleInputTransformation() {
        testWrapper(
                3,
                sources -> {
                    MultipleInputTransformation<Integer> multipleInputTransformation =
                            new MultipleInputTransformation<>(
                                    "mul",
                                    new MultipleInputAllRoundWrapperOperatorTest
                                            .LifeCycleTrackingTwoInputStreamOperatorFactory(),
                                    BasicTypeInfo.INT_TYPE_INFO,
                                    4);
                    sources.forEach(
                            s -> multipleInputTransformation.addInput(s.getTransformation()));
                    sources.get(0)
                            .getExecutionEnvironment()
                            .addOperator(multipleInputTransformation);
                    new MultipleConnectedStreams(sources.get(0).getExecutionEnvironment())
                            .transform(multipleInputTransformation)
                            .map(x -> x);
                });
    }

    @Test
    public void testKeyedMultipleInputTransformation() {
        testWrapper(
                3,
                sources -> {
                    MultipleInputTransformation<Integer> multipleInputTransformation =
                            new MultipleInputTransformation<>(
                                    "mul",
                                    new MultipleInputAllRoundWrapperOperatorTest
                                            .LifeCycleTrackingTwoInputStreamOperatorFactory(),
                                    BasicTypeInfo.INT_TYPE_INFO,
                                    4);
                    sources.forEach(
                            s ->
                                    multipleInputTransformation.addInput(
                                            s.keyBy(i -> i).getTransformation()));
                    sources.get(0)
                            .getExecutionEnvironment()
                            .addOperator(multipleInputTransformation);
                    new MultipleConnectedStreams(sources.get(0).getExecutionEnvironment())
                            .transform(multipleInputTransformation)
                            .map(x -> x);
                });
    }

    private void testWrapper(
            int numberOfSources, Consumer<List<DataStream<Integer>>> graphBuilder) {
        StreamExecutionEnvironment env = new StreamExecutionEnvironment();
        DraftExecutionEnvironment draftEnv =
                new DraftExecutionEnvironment(env, getOperatorWrapper());
        List<DataStream<Integer>> sources = new ArrayList<>();
        for (int i = 0; i < numberOfSources; ++i) {
            DataStreamSource<Integer> source =
                    env.addSource(new DraftExecutionEnvironment.EmptySource<>());
            source.returns(BasicTypeInfo.INT_TYPE_INFO);
            env.addOperator(source.getTransformation());

            sources.add(draftEnv.addDraftSource(source, BasicTypeInfo.INT_TYPE_INFO));
        }
        graphBuilder.accept(sources);
        draftEnv.copyToActualEnvironment();

        StreamGraph wrappedGraph = env.getStreamGraph();
        StreamGraph nonWrappedGraph = draftEnv.getStreamGraph();

        checkWrappedGraph(nonWrappedGraph, wrappedGraph, draftEnv);
    }
}
