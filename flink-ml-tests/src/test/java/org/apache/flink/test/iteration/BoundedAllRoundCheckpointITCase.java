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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.test.iteration.operators.EpochRecord;
import org.apache.flink.test.iteration.operators.FailingMap;
import org.apache.flink.test.iteration.operators.IncrementEpochMap;
import org.apache.flink.test.iteration.operators.OutputRecord;
import org.apache.flink.test.iteration.operators.SequenceSource;
import org.apache.flink.test.iteration.operators.TwoInputReduceAllRoundProcessFunction;
import org.apache.flink.testutils.junit.SharedObjects;
import org.apache.flink.testutils.junit.SharedReference;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.TestLogger;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.createMiniClusterConfiguration;
import static org.junit.Assert.assertEquals;

/** Tests checkpoints. */
@RunWith(Parameterized.class)
public class BoundedAllRoundCheckpointITCase extends TestLogger {

    @Rule public final SharedObjects sharedObjects = SharedObjects.create();

    private SharedReference<List<OutputRecord<Integer>>> result;

    @Parameterized.Parameter(0)
    public int failoverCount;

    @Parameterized.Parameter(1)
    public boolean sync;

    @Parameterized.Parameters(name = "failoverCount = {0}, sync = {1}")
    public static Collection<Object[]> params() {
        int[] failoverCounts = {1000, 4000, 8000, 15900};
        boolean[] syncs = {true, false};

        List<Object[]> result = new ArrayList<>();
        for (int failoverCount : failoverCounts) {
            for (boolean sync : syncs) {
                result.add(new Object[] {failoverCount, sync});
            }
        }

        return result;
    }

    @Before
    public void setup() {
        result = sharedObjects.add(new ArrayList<>());
    }

    @Test
    public void testFailoverAndRestore() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2))) {
            miniCluster.start();

            // Create the test job
            JobGraph jobGraph =
                    createVariableAndConstantJobGraph(
                            4, 1000, false, 0, sync, 4, failoverCount, new CollectSink(result));
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
            assertEquals(6, roundsStat.size());
            for (int i = 0; i <= 4; ++i) {
                // In this case we could only check the final result, the number of records is not
                // deterministic.
                assertEquals(4 * (0 + 999) * 1000 / 2, (int) roundsStat.get(i).f1);
            }
        }
    }

    static JobGraph createVariableAndConstantJobGraph(
            int numSources,
            int numRecordsPerSource,
            boolean holdSource,
            int period,
            boolean sync,
            int maxRound,
            int failoverCount,
            SinkFunction<OutputRecord<Integer>> sinkFunction) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.enableCheckpointing(500, CheckpointingMode.EXACTLY_ONCE);
        env.setParallelism(1);
        DataStream<EpochRecord> variableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<EpochRecord>() {})
                        .setParallelism(numSources)
                        .name("Variable");
        DataStream<EpochRecord> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, holdSource, period))
                        .setParallelism(numSources)
                        .name("Constant");
        DataStreamList outputs =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(variableSource),
                        DataStreamList.of(constSource),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<EpochRecord> reducer =
                                    variableStreams
                                            .<EpochRecord>get(0)
                                            .connect(dataStreams.<EpochRecord>get(0))
                                            .process(
                                                    new TwoInputReduceAllRoundProcessFunction(
                                                            sync, maxRound));
                            DataStream<EpochRecord> failedMap =
                                    reducer.map(new FailingMap(failoverCount) {});
                            return new IterationBodyResult(
                                    DataStreamList.of(
                                            failedMap
                                                    .map(new IncrementEpochMap())
                                                    .setParallelism(numSources)),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    new OutputTag<OutputRecord<Integer>>(
                                                            "output") {})));
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
