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

package org.apache.flink.test.iteration;

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.test.iteration.operators.EpochRecord;
import org.apache.flink.test.iteration.operators.FailingMap;
import org.apache.flink.test.iteration.operators.OutputRecord;
import org.apache.flink.test.iteration.operators.SequenceSource;
import org.apache.flink.test.iteration.operators.TwoInputReducePerRoundOperator;
import org.apache.flink.testutils.junit.SharedObjects;
import org.apache.flink.testutils.junit.SharedReference;
import org.apache.flink.util.TestLogger;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.createMiniClusterConfiguration;
import static org.junit.Assert.assertEquals;

/** Tests checkpoints. */
@RunWith(Parameterized.class)
public class BoundedPerRoundCheckpointITCase extends TestLogger {

    @Rule public final SharedObjects sharedObjects = SharedObjects.create();

    private SharedReference<List<OutputRecord<Integer>>> result;

    @Parameterized.Parameter(0)
    public int failoverCount;

    @Parameterized.Parameters(name = "failoverCount = {0}")
    public static Collection<Object[]> params() {
        return Arrays.asList(
                new Object[] {1000},
                new Object[] {4000},
                new Object[] {6123},
                new Object[] {8000},
                new Object[] {10875},
                new Object[] {15900});
    }

    @Before
    public void setup() {
        result = sharedObjects.add(new ArrayList<>());
    }

    @Test
    public void testFailoverAndRestore() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2))) {
            miniCluster.start();

            // Creates the test job
            JobGraph jobGraph =
                    createVariableAndConstantJobGraph(
                            4, 1000, 4, failoverCount, new CollectSink(result));
            miniCluster.executeJobBlocking(jobGraph);

            Map<Integer, Tuple2<Integer, Integer>> roundsStat = new HashMap<>();
            for (OutputRecord<Integer> output : result.get()) {
                Tuple2<Integer, Integer> state =
                        roundsStat.computeIfAbsent(
                                output.getRound(), ignored -> new Tuple2<>(0, 0));
                state.f0++;
                state.f1 = output.getValue();
            }

            // 0 ~ 4 round and termination information
            assertEquals(4, roundsStat.size());
            for (int i = 0; i < 4; ++i) {
                // In this case we could only check the final result, the number of records is not
                // deterministic.
                assertEquals(4 * (0 + 999) * 1000 / 2, (int) roundsStat.get(i).f1);
            }
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static JobGraph createVariableAndConstantJobGraph(
            int numSources,
            int numRecordsPerSource,
            int maxRound,
            int failoverCount,
            SinkFunction<OutputRecord<Integer>> sinkFunction) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.enableCheckpointing(500, CheckpointingMode.EXACTLY_ONCE);
        env.setParallelism(1);
        DataStream<Integer> variableSource =
                env.addSource(new SequenceSource(1, false, 0))
                        .setParallelism(1)
                        .map(EpochRecord::getValue)
                        .setParallelism(1)
                        .name("Variable");
        DataStream<Integer> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, false, 0))
                        .setParallelism(numSources)
                        .map(EpochRecord::getValue)
                        .setParallelism(numSources)
                        .name("Constant");
        DataStreamList outputs =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(variableSource),
                        ReplayableDataStreamList.replay(constSource),
                        IterationConfig.newBuilder().build(),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<Integer> reducer =
                                    (SingleOutputStreamOperator)
                                            IterationBody.forEachRound(
                                                            DataStreamList.of(
                                                                    variableStreams.get(0),
                                                                    dataStreams
                                                                            .<Integer>get(0)
                                                                            .map(
                                                                                    new FailingMap<
                                                                                            Integer>(
                                                                                            failoverCount) {})),
                                                            (streams) -> {
                                                                DataStream<Integer> variableStream =
                                                                        streams.get(0);
                                                                DataStream<Integer> constStream =
                                                                        streams.get(1);
                                                                return DataStreamList.of(
                                                                        variableStream
                                                                                .connect(
                                                                                        constStream)
                                                                                .transform(
                                                                                        "Reducer",
                                                                                        BasicTypeInfo
                                                                                                .INT_TYPE_INFO,
                                                                                        new TwoInputReducePerRoundOperator())
                                                                                .setParallelism(1));
                                                            })
                                                    .get(0);

                            return new IterationBodyResult(
                                    DataStreamList.of(reducer.filter(x -> x < maxRound)),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    TwoInputReducePerRoundOperator.OUTPUT_TAG)),
                                    reducer.filter(x -> x < maxRound).setParallelism(1));
                        });
        outputs.<OutputRecord<Integer>>get(0).addSink(sinkFunction);

        return env.getStreamGraph().getJobGraph();
    }

    private static class CollectSink implements SinkFunction<OutputRecord<Integer>> {

        private final SharedReference<List<OutputRecord<Integer>>> result;

        private CollectSink(SharedReference<List<OutputRecord<Integer>>> result) {
            this.result = result;
        }

        @Override
        public void invoke(OutputRecord<Integer> value, Context context) throws Exception {
            result.get().add(value);
        }
    }
}
