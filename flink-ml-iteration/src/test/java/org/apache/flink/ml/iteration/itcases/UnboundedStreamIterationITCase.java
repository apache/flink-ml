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

package org.apache.flink.ml.iteration.itcases;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.ml.iteration.DataStreamList;
import org.apache.flink.ml.iteration.IterationBodyResult;
import org.apache.flink.ml.iteration.Iterations;
import org.apache.flink.ml.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.ml.iteration.itcases.operators.OutputRecord;
import org.apache.flink.ml.iteration.itcases.operators.ReduceAllRoundProcessFunction;
import org.apache.flink.ml.iteration.itcases.operators.SequenceSource;
import org.apache.flink.ml.iteration.itcases.operators.TwoInputReduceAllRoundProcessFunction;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.util.OutputTag;

import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import static org.junit.Assert.assertEquals;

/** Integration cases for unbounded iteration */
public class UnboundedStreamIterationITCase {

    private static BlockingQueue<OutputRecord<Integer>> result = new LinkedBlockingQueue<>();

    @Before
    public void setup() {
        result.clear();
    }

    @Test(timeout = 60000)
    public void testVariableOnlyUnboundedIteration() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2))) {
            miniCluster.start();

            // Create the test job
            JobGraph jobGraph =
                    createVariableOnlyJobGraph(
                            4,
                            1000,
                            true,
                            0,
                            false,
                            1,
                            new SinkFunction<OutputRecord<Integer>>() {
                                @Override
                                public void invoke(OutputRecord<Integer> value, Context context) {
                                    result.add(value);
                                }
                            });
            miniCluster.submitJob(jobGraph);

            int expectedOutputs = 2 * 4000;
            Map<Integer, Tuple2<Integer, Integer>> roundsStat = new HashMap<>();
            for (int i = 0; i < expectedOutputs; ++i) {
                OutputRecord<Integer> next = result.take();
                assertEquals(OutputRecord.Event.PROCESS_ELEMENT, next.getEvent());
                Tuple2<Integer, Integer> state =
                        roundsStat.computeIfAbsent(next.getRound(), ignored -> new Tuple2<>(0, 0));
                state.f0++;
                state.f1 = next.getValue();
            }

            verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
        }
    }

    @Test(timeout = 60000)
    public void testVariableOnlyBoundedIteration() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2))) {
            miniCluster.start();

            // Create the test job
            JobGraph jobGraph =
                    createVariableOnlyJobGraph(
                            4,
                            1000,
                            false,
                            0,
                            false,
                            1,
                            new SinkFunction<OutputRecord<Integer>>() {
                                @Override
                                public void invoke(OutputRecord<Integer> value, Context context) {
                                    result.add(value);
                                }
                            });
            miniCluster.executeJobBlocking(jobGraph);

            assertEquals(8001, result.size());

            Map<Integer, Tuple2<Integer, Integer>> roundsStat = new HashMap<>();
            for (int i = 0; i < 8000; ++i) {
                OutputRecord<Integer> next = result.take();
                assertEquals(OutputRecord.Event.PROCESS_ELEMENT, next.getEvent());
                Tuple2<Integer, Integer> state =
                        roundsStat.computeIfAbsent(next.getRound(), ignored -> new Tuple2<>(0, 0));
                state.f0++;
                state.f1 = next.getValue();
            }

            verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
            assertEquals(OutputRecord.Event.TERMINATED, result.take().getEvent());
        }
    }

    @Test(timeout = 60000)
    public void testVariableAndConstantsUnboundedIteration() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2))) {
            miniCluster.start();

            // Create the test job
            JobGraph jobGraph =
                    createVariableAndConstantJobGraph(
                            4,
                            1000,
                            true,
                            0,
                            false,
                            1,
                            new SinkFunction<OutputRecord<Integer>>() {
                                @Override
                                public void invoke(OutputRecord<Integer> value, Context context) {
                                    result.add(value);
                                }
                            });
            miniCluster.submitJob(jobGraph);

            int expectedOutputs = 2 * 4000;
            Map<Integer, Tuple2<Integer, Integer>> roundsStat = new HashMap<>();
            for (int i = 0; i < expectedOutputs; ++i) {
                OutputRecord<Integer> next = result.take();
                assertEquals(OutputRecord.Event.PROCESS_ELEMENT, next.getEvent());
                Tuple2<Integer, Integer> state =
                        roundsStat.computeIfAbsent(next.getRound(), ignored -> new Tuple2<>(0, 0));
                state.f0++;
                state.f1 = next.getValue();
            }

            verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
        }
    }

    @Test(timeout = 60000)
    public void testVariableAndConstantBoundedIteration() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2))) {
            miniCluster.start();

            // Create the test job
            JobGraph jobGraph =
                    createVariableAndConstantJobGraph(
                            4,
                            1000,
                            false,
                            0,
                            false,
                            1,
                            new SinkFunction<OutputRecord<Integer>>() {
                                @Override
                                public void invoke(OutputRecord<Integer> value, Context context) {
                                    result.add(value);
                                }
                            });
            miniCluster.executeJobBlocking(jobGraph);

            assertEquals(8001, result.size());

            Map<Integer, Tuple2<Integer, Integer>> roundsStat = new HashMap<>();
            for (int i = 0; i < 8000; ++i) {
                OutputRecord<Integer> next = result.take();
                assertEquals(OutputRecord.Event.PROCESS_ELEMENT, next.getEvent());
                Tuple2<Integer, Integer> state =
                        roundsStat.computeIfAbsent(next.getRound(), ignored -> new Tuple2<>(0, 0));
                state.f0++;
                state.f1 = next.getValue();
            }

            verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
            assertEquals(OutputRecord.Event.TERMINATED, result.take().getEvent());
        }
    }

    static MiniClusterConfiguration createMiniClusterConfiguration(int numTm, int numSlot) {
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18081);
        return new MiniClusterConfiguration.Builder()
                .setConfiguration(configuration)
                .setNumTaskManagers(numTm)
                .setNumSlotsPerTaskManager(numSlot)
                .build();
    }

    static JobGraph createVariableOnlyJobGraph(
            int numSources,
            int numRecordsPerSource,
            boolean holdSource,
            int period,
            boolean sync,
            int maxRound,
            SinkFunction<OutputRecord<Integer>> sinkFunction) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        DataStream<Integer> source =
                env.addSource(new SequenceSource(numRecordsPerSource, holdSource, period))
                        .setParallelism(numSources);
        DataStreamList outputs =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(source),
                        DataStreamList.of(),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<Integer> reducer =
                                    variableStreams
                                            .<Integer>get(0)
                                            .process(
                                                    new ReduceAllRoundProcessFunction(
                                                            sync, maxRound));
                            return new IterationBodyResult(
                                    DataStreamList.of(
                                            reducer.map(x -> x).setParallelism(numSources)),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    new OutputTag<OutputRecord<Integer>>(
                                                            "output") {})));
                        });
        outputs.<OutputRecord<Integer>>get(0).addSink(sinkFunction);

        return env.getStreamGraph().getJobGraph();
    }

    static JobGraph createVariableAndConstantJobGraph(
            int numSources,
            int numRecordsPerSource,
            boolean holdSource,
            int period,
            boolean sync,
            int maxRound,
            SinkFunction<OutputRecord<Integer>> sinkFunction) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        DataStream<Integer> variableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {})
                        .setParallelism(numSources)
                        .name("Variable");
        DataStream<Integer> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, holdSource, period))
                        .setParallelism(numSources)
                        .name("Constants");
        DataStreamList outputs =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(variableSource),
                        DataStreamList.of(constSource),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<Integer> reducer =
                                    variableStreams
                                            .<Integer>get(0)
                                            .connect(dataStreams.<Integer>get(0))
                                            .process(
                                                    new TwoInputReduceAllRoundProcessFunction(
                                                            sync, maxRound));
                            return new IterationBodyResult(
                                    DataStreamList.of(
                                            reducer.map(x -> x).setParallelism(numSources)),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    new OutputTag<OutputRecord<Integer>>(
                                                            "output") {})));
                        });
        outputs.<OutputRecord<Integer>>get(0).addSink(sinkFunction);

        return env.getStreamGraph().getJobGraph();
    }

    static void verifyResult(
            Map<Integer, Tuple2<Integer, Integer>> roundsStat,
            int expectedRound,
            int recordsEachRound,
            int valueEachRound) {
        assertEquals(expectedRound, roundsStat.size());
        for (int i = 0; i < expectedRound; ++i) {
            assertEquals(recordsEachRound, (int) roundsStat.get(i).f0);
            assertEquals(valueEachRound, (int) roundsStat.get(i).f1);
        }
    }
}
