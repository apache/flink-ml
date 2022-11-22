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

import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.jobgraph.JobVertex;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.streaming.api.functions.sink.DiscardingSink;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.TestLogger;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;

/** Verifies the created job graph satisfy the expectation. */
public class IterationConstructionTest extends TestLogger {

    @Test
    public void testEmptyIterationBody() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(4);
        DataStream<Integer> variableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .name("Variable");
        DataStreamList result =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(variableSource),
                        DataStreamList.of(),
                        ((variableStreams, dataStreams) ->
                                new IterationBodyResult(variableStreams, dataStreams)));

        JobGraph jobGraph = env.getStreamGraph().getJobGraph();

        List<String> expectedVertexNames =
                Arrays.asList(
                        /* 0 */ "Source: Variable -> input-Variable",
                        /* 1 */ "head-Variable",
                        /* 2 */ "tail-head-Variable -> filter-tail");
        List<Integer> expectedParallelisms = Arrays.asList(4, 4, 4);

        List<JobVertex> vertices = jobGraph.getVerticesSortedTopologicallyFromSources();
        assertEquals(
                expectedVertexNames,
                vertices.stream().map(JobVertex::getName).collect(Collectors.toList()));
        assertEquals(
                expectedParallelisms,
                vertices.stream().map(JobVertex::getParallelism).collect(Collectors.toList()));
        assertNotNull(vertices.get(1).getCoLocationGroup());
        assertNotNull(vertices.get(2).getCoLocationGroup());
        assertSame(vertices.get(1).getCoLocationGroup(), vertices.get(2).getCoLocationGroup());
    }

    @Test
    public void testNotChainingHeadOperator() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(4);
        DataStream<Integer> variableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .name("Variable")
                        .map(x -> x)
                        .name("map")
                        .setParallelism(2);
        DataStreamList result =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(variableSource),
                        DataStreamList.of(),
                        ((variableStreams, dataStreams) ->
                                new IterationBodyResult(variableStreams, dataStreams)));

        JobGraph jobGraph = env.getStreamGraph().getJobGraph();

        List<String> expectedVertexNames =
                Arrays.asList(
                        /* 0 */ "Source: Variable",
                        /* 1 */ "map -> input-map",
                        /* 2 */ "head-map",
                        /* 3 */ "tail-head-map -> filter-tail");
        List<Integer> expectedParallelisms = Arrays.asList(4, 2, 2, 2);

        List<JobVertex> vertices = jobGraph.getVerticesSortedTopologicallyFromSources();
        assertEquals(
                expectedVertexNames,
                vertices.stream().map(JobVertex::getName).collect(Collectors.toList()));
        assertEquals(
                expectedParallelisms,
                vertices.stream().map(JobVertex::getParallelism).collect(Collectors.toList()));
        assertNotNull(vertices.get(2).getCoLocationGroup());
        assertSame(vertices.get(2).getCoLocationGroup(), vertices.get(3).getCoLocationGroup());
    }

    @Test
    public void testUnboundedIteration() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        DataStream<Integer> variableSource1 =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(2)
                        .name("Variable0");
        DataStream<Integer> variableSource2 =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(3)
                        .name("Variable1");

        DataStream<Integer> constantSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(3)
                        .name("Constant");

        DataStreamList result =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(variableSource1, variableSource2),
                        DataStreamList.of(constantSource),
                        new IterationBody() {

                            @Override
                            public IterationBodyResult process(
                                    DataStreamList variableStreams, DataStreamList dataStreams) {
                                SingleOutputStreamOperator<Integer> processor =
                                        variableStreams
                                                .<Integer>get(0)
                                                .union(variableStreams.<Integer>get(1))
                                                .connect(dataStreams.<Integer>get(0))
                                                .process(
                                                        new CoProcessFunction<
                                                                Integer, Integer, Integer>() {
                                                            @Override
                                                            public void processElement1(
                                                                    Integer value,
                                                                    Context ctx,
                                                                    Collector<Integer> out)
                                                                    throws Exception {}

                                                            @Override
                                                            public void processElement2(
                                                                    Integer value,
                                                                    Context ctx,
                                                                    Collector<Integer> out)
                                                                    throws Exception {}
                                                        })
                                                .name("Processor")
                                                .setParallelism(4);

                                return new IterationBodyResult(
                                        DataStreamList.of(
                                                processor
                                                        .map(x -> x)
                                                        .name("Feedback0")
                                                        .setParallelism(2),
                                                processor
                                                        .map(x -> x)
                                                        .name("Feedback1")
                                                        .setParallelism(3)),
                                        DataStreamList.of(
                                                processor.getSideOutput(
                                                        new OutputTag<Integer>("output") {})));
                            }
                        });
        result.get(0).addSink(new DiscardingSink<>()).name("Sink").setParallelism(4);

        List<String> expectedVertexNames =
                Arrays.asList(
                        /* 0 */ "Source: Variable0 -> input-Variable0",
                        /* 1 */ "Source: Variable1 -> input-Variable1",
                        /* 2 */ "Source: Constant -> input-Constant",
                        /* 3 */ "head-Variable0",
                        /* 4 */ "head-Variable1",
                        /* 5 */ "Processor",
                        /* 6 */ "Feedback0",
                        /* 7 */ "tail-Feedback0 -> filter-tail",
                        /* 8 */ "Feedback1",
                        /* 9 */ "tail-Feedback1 -> filter-tail",
                        /* 10 */ "tail-map-SideOutput",
                        /* 11 */ "output-SideOutput -> Sink: Sink");
        List<Integer> expectedParallelisms = Arrays.asList(2, 3, 3, 2, 3, 4, 2, 2, 3, 3, 1, 4);

        JobGraph jobGraph = env.getStreamGraph().getJobGraph();
        List<JobVertex> vertices = jobGraph.getVerticesSortedTopologicallyFromSources();
        assertEquals(
                expectedVertexNames,
                vertices.stream().map(JobVertex::getName).collect(Collectors.toList()));
        assertEquals(
                expectedParallelisms,
                vertices.stream().map(JobVertex::getParallelism).collect(Collectors.toList()));

        assertNotNull(vertices.get(3).getCoLocationGroup());
        assertNotNull(vertices.get(4).getCoLocationGroup());
        assertSame(vertices.get(3).getCoLocationGroup(), vertices.get(7).getCoLocationGroup());
        assertSame(vertices.get(4).getCoLocationGroup(), vertices.get(9).getCoLocationGroup());
    }

    @Test
    public void testBoundedIterationWithTerminationCriteria() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        DataStream<Integer> variableSource1 =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(2)
                        .name("Variable0");
        DataStream<Integer> variableSource2 =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(3)
                        .name("Variable1");

        DataStream<Integer> constantSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(3)
                        .name("Constant");

        DataStreamList result =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(variableSource1, variableSource2),
                        ReplayableDataStreamList.notReplay(constantSource),
                        IterationConfig.newBuilder().build(),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<Integer> processor =
                                    variableStreams
                                            .<Integer>get(0)
                                            .union(variableStreams.<Integer>get(1))
                                            .connect(dataStreams.<Integer>get(0))
                                            .process(
                                                    new CoProcessFunction<
                                                            Integer, Integer, Integer>() {
                                                        @Override
                                                        public void processElement1(
                                                                Integer value,
                                                                Context ctx,
                                                                Collector<Integer> out)
                                                                throws Exception {}

                                                        @Override
                                                        public void processElement2(
                                                                Integer value,
                                                                Context ctx,
                                                                Collector<Integer> out)
                                                                throws Exception {}
                                                    })
                                            .name("Processor")
                                            .setParallelism(4);

                            return new IterationBodyResult(
                                    DataStreamList.of(
                                            processor
                                                    .map(x -> x)
                                                    .name("Feedback0")
                                                    .setParallelism(2),
                                            processor
                                                    .map(x -> x)
                                                    .name("Feedback1")
                                                    .setParallelism(3)),
                                    DataStreamList.of(
                                            processor.getSideOutput(
                                                    new OutputTag<Integer>("output") {})),
                                    processor.map(x -> x).name("Termination").setParallelism(5));
                        });
        result.get(0).addSink(new DiscardingSink<>()).name("Sink").setParallelism(4);

        List<String> expectedVertexNames =
                Arrays.asList(
                        /* 0 */ "Source: Variable0 -> input-Variable0",
                        /* 1 */ "Source: Variable1 -> input-Variable1",
                        /* 2 */ "Source: Constant -> input-Constant",
                        /* 3 */ "Source: Termination -> input-Termination",
                        /* 4 */ "head-Variable0",
                        /* 5 */ "head-Variable1",
                        /* 6 */ "Processor",
                        /* 7 */ "Feedback0",
                        /* 8 */ "tail-Feedback0 -> filter-tail",
                        /* 9 */ "Feedback1",
                        /* 10 */ "tail-Feedback1 -> filter-tail",
                        /* 11 */ "Termination",
                        /* 12 */ "head-Termination",
                        /* 13 */ "criteria-merge",
                        /* 14 */ "tail-criteria-merge -> filter-tail",
                        /* 15 */ "tail-map-SideOutput",
                        /* 16 */ "output-SideOutput -> Sink: Sink");
        List<Integer> expectedParallelisms =
                Arrays.asList(2, 3, 3, 5, 2, 3, 4, 2, 2, 3, 3, 5, 5, 5, 5, 1, 4);

        JobGraph jobGraph = env.getStreamGraph().getJobGraph();
        List<JobVertex> vertices = jobGraph.getVerticesSortedTopologicallyFromSources();
        assertEquals(
                expectedVertexNames,
                vertices.stream().map(JobVertex::getName).collect(Collectors.toList()));
        assertEquals(
                expectedParallelisms,
                vertices.stream().map(JobVertex::getParallelism).collect(Collectors.toList()));

        assertNotNull(vertices.get(4).getCoLocationGroup());
        assertNotNull(vertices.get(5).getCoLocationGroup());
        assertNotNull(vertices.get(12).getCoLocationGroup());
        assertSame(vertices.get(4).getCoLocationGroup(), vertices.get(8).getCoLocationGroup());
        assertSame(vertices.get(5).getCoLocationGroup(), vertices.get(10).getCoLocationGroup());
        assertSame(vertices.get(12).getCoLocationGroup(), vertices.get(14).getCoLocationGroup());
    }

    @Test
    public void testReplayedIteration() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        DataStream<Integer> variableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(2)
                        .name("Variable");

        DataStream<Integer> constantSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(3)
                        .name("Constant");

        DataStreamList result =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(variableSource),
                        ReplayableDataStreamList.replay(constantSource),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(IterationConfig.OperatorLifeCycle.PER_ROUND)
                                .build(),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<Integer> processor =
                                    variableStreams
                                            .<Integer>get(0)
                                            .connect(dataStreams.<Integer>get(0))
                                            .process(
                                                    new CoProcessFunction<
                                                            Integer, Integer, Integer>() {
                                                        @Override
                                                        public void processElement1(
                                                                Integer value,
                                                                Context ctx,
                                                                Collector<Integer> out)
                                                                throws Exception {}

                                                        @Override
                                                        public void processElement2(
                                                                Integer value,
                                                                Context ctx,
                                                                Collector<Integer> out)
                                                                throws Exception {}
                                                    })
                                            .name("Processor")
                                            .setParallelism(4);

                            return new IterationBodyResult(
                                    DataStreamList.of(
                                            processor
                                                    .map(x -> x)
                                                    .name("Feedback")
                                                    .setParallelism(2)),
                                    DataStreamList.of(
                                            processor.getSideOutput(
                                                    new OutputTag<Integer>("output") {})),
                                    processor.map(x -> x).name("Termination").setParallelism(5));
                        });
        result.get(0).addSink(new DiscardingSink<>()).name("Sink").setParallelism(4);

        List<String> expectedVertexNames =
                Arrays.asList(
                        /* 0 */ "Source: Variable -> input-Variable",
                        /* 1 */ "Source: Constant -> input-Constant",
                        /* 2 */ "Source: Termination -> input-Termination",
                        /* 3 */ "head-Variable",
                        /* 4 */ "Replayer-Constant",
                        /* 5 */ "Processor",
                        /* 6 */ "Feedback",
                        /* 7 */ "tail-Feedback -> filter-tail",
                        /* 8 */ "Termination",
                        /* 9 */ "head-Termination",
                        /* 10 */ "criteria-merge",
                        /* 11 */ "tail-criteria-merge -> filter-tail",
                        /* 12 */ "tail-map-SideOutput",
                        /* 13 */ "output-SideOutput -> Sink: Sink");
        List<Integer> expectedParallelisms =
                Arrays.asList(2, 3, 5, 2, 3, 4, 2, 2, 5, 5, 5, 5, 1, 4);

        JobGraph jobGraph = env.getStreamGraph().getJobGraph();
        List<JobVertex> vertices = jobGraph.getVerticesSortedTopologicallyFromSources();
        assertEquals(
                expectedVertexNames,
                vertices.stream().map(JobVertex::getName).collect(Collectors.toList()));
        assertEquals(
                expectedParallelisms,
                vertices.stream().map(JobVertex::getParallelism).collect(Collectors.toList()));

        assertNotNull(vertices.get(3).getCoLocationGroup());
        assertNotNull(vertices.get(9).getCoLocationGroup());
        assertSame(vertices.get(3).getCoLocationGroup(), vertices.get(7).getCoLocationGroup());
        assertSame(vertices.get(9).getCoLocationGroup(), vertices.get(11).getCoLocationGroup());
    }
}
