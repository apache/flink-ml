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

package org.apache.flink.iteration.compile;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.iteration.operator.OperatorWrapper;
import org.apache.flink.iteration.operator.WrapperOperatorFactory;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.graph.StreamGraph;
import org.apache.flink.streaming.api.graph.StreamNode;
import org.apache.flink.streaming.api.operators.StreamFilter;
import org.apache.flink.streaming.api.operators.StreamMap;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.partitioner.StreamPartitioner;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.TestLogger;

import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/** Tests switching the operator wrapper during creating the draft. */
public class DraftExecutionEnvironmentSwitchWrapperTest extends TestLogger {

    @Test
    public void testSwitchingOperatorWrappers() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.getConfig().disableGenericTypes();
        DataStream<Integer> source =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {});

        DraftExecutionEnvironment draftEnv =
                new DraftExecutionEnvironment(env, new FirstWrapper<>());
        DataStream<Integer> draftSource =
                draftEnv.addDraftSource(source, BasicTypeInfo.INT_TYPE_INFO);

        DataStream<Integer> firstPart =
                draftSource.process(
                        new ProcessFunction<Integer, Integer>() {
                            @Override
                            public void processElement(
                                    Integer value, Context ctx, Collector<Integer> out) {}
                        });
        draftEnv.setCurrentWrapper(new SecondWrapper<>());
        firstPart
                .keyBy(x -> x)
                .process(
                        new KeyedProcessFunction<Integer, Integer, Integer>() {
                            @Override
                            public void processElement(
                                    Integer value, Context ctx, Collector<Integer> out) {}
                        });

        draftEnv.copyToActualEnvironment();
        StreamGraph graph = env.getStreamGraph();

        List<Integer> nodeIds =
                graph.getStreamNodes().stream()
                        .filter(node -> node.getInEdges().size() > 0)
                        .map(StreamNode::getId)
                        .sorted()
                        .collect(Collectors.toList());
        assertEquals(2, nodeIds.size());

        assertEquals(
                FirstWrapper.class,
                ((WrapperOperatorFactory<?>)
                                graph.getStreamNode(nodeIds.get(0)).getOperatorFactory())
                        .getWrapper()
                        .getClass());
        assertEquals(
                SecondWrapper.class,
                ((WrapperOperatorFactory<?>)
                                graph.getStreamNode(nodeIds.get(1)).getOperatorFactory())
                        .getWrapper()
                        .getClass());
    }

    private static class FirstWrapper<T> implements OperatorWrapper<T, T> {
        @Override
        public StreamOperator<T> wrap(
                StreamOperatorParameters<T> operatorParameters,
                StreamOperatorFactory<T> operatorFactory) {
            return new StreamMap<>((MapFunction<T, T>) value -> value);
        }

        @Override
        public Class<? extends StreamOperator> getStreamOperatorClass(
                ClassLoader classLoader, StreamOperatorFactory<T> operatorFactory) {
            return StreamMap.class;
        }

        @Override
        public <KEY> KeySelector<T, KEY> wrapKeySelector(KeySelector<T, KEY> keySelector) {
            return keySelector;
        }

        @Override
        public StreamPartitioner<T> wrapStreamPartitioner(StreamPartitioner<T> streamPartitioner) {
            return streamPartitioner;
        }

        @Override
        public OutputTag<T> wrapOutputTag(OutputTag<T> outputTag) {
            return outputTag;
        }

        @Override
        public TypeInformation<T> getWrappedTypeInfo(TypeInformation<T> typeInfo) {
            return typeInfo;
        }
    }

    private static class SecondWrapper<T> implements OperatorWrapper<T, T> {
        @Override
        public StreamOperator<T> wrap(
                StreamOperatorParameters<T> operatorParameters,
                StreamOperatorFactory<T> operatorFactory) {
            return new StreamFilter<>((FilterFunction<T>) value -> true);
        }

        @Override
        public Class<? extends StreamOperator> getStreamOperatorClass(
                ClassLoader classLoader, StreamOperatorFactory<T> operatorFactory) {
            return StreamFilter.class;
        }

        @Override
        public <KEY> KeySelector<T, KEY> wrapKeySelector(KeySelector<T, KEY> keySelector) {
            return keySelector;
        }

        @Override
        public StreamPartitioner<T> wrapStreamPartitioner(StreamPartitioner<T> streamPartitioner) {
            return streamPartitioner;
        }

        @Override
        public OutputTag<T> wrapOutputTag(OutputTag<T> outputTag) {
            return outputTag;
        }

        @Override
        public TypeInformation<T> getWrappedTypeInfo(TypeInformation<T> typeInfo) {
            return typeInfo;
        }
    }
}
