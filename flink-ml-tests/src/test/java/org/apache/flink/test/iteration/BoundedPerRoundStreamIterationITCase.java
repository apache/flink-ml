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
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.test.iteration.operators.EpochRecord;
import org.apache.flink.test.iteration.operators.OutputRecord;
import org.apache.flink.test.iteration.operators.SequenceSource;
import org.apache.flink.test.iteration.operators.TwoInputReducePerRoundOperator;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.verifyResult;
import static org.junit.Assert.assertEquals;

/** Tests the per-round iterations. */
public class BoundedPerRoundStreamIterationITCase {

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    private static BlockingQueue<OutputRecord<Integer>> result = new LinkedBlockingQueue<>();

    @Before
    public void setup() {
        result.clear();
    }

    @Test
    public void testPerRoundIteration() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2))) {
            miniCluster.start();

            JobGraph jobGraph =
                    createPerRoundJobGraph(
                            4,
                            1000,
                            5,
                            new SinkFunction<OutputRecord<Integer>>() {
                                @Override
                                public void invoke(OutputRecord<Integer> value, Context context) {
                                    result.add(value);
                                }
                            });
            miniCluster.executeJobBlocking(jobGraph);

            assertEquals(5, result.size());

            Map<Integer, Tuple2<Integer, Integer>> roundsStat = new HashMap<>();
            for (int i = 0; i < 5; ++i) {
                OutputRecord<Integer> next = result.take();
                assertEquals(OutputRecord.Event.TERMINATED, next.getEvent());
                Tuple2<Integer, Integer> state =
                        roundsStat.computeIfAbsent(next.getRound(), ignored -> new Tuple2<>(0, 0));
                state.f0++;
                state.f1 = next.getValue();
            }

            verifyResult(roundsStat, 5, 1, 4 * (0 + 999) * 1000 / 2);
        }
    }

    private MiniClusterConfiguration createMiniClusterConfiguration(int numTm, int numSlot)
            throws IOException {
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18081);
        configuration.set(
                IterationOptions.DATA_CACHE_PATH,
                "file://" + tempFolder.newFolder().getAbsolutePath());
        return new MiniClusterConfiguration.Builder()
                .setConfiguration(configuration)
                .setNumTaskManagers(numTm)
                .setNumSlotsPerTaskManager(numSlot)
                .build();
    }

    private static JobGraph createPerRoundJobGraph(
            int numSources,
            int numRecordsPerSource,
            int maxRound,
            SinkFunction<OutputRecord<Integer>> sinkFunction) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataStream<Integer> variableSource = env.fromElements(0);
        DataStream<EpochRecord> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, false, 0))
                        .setParallelism(numSources)
                        .name("Constants");

        DataStreamList outputs =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(variableSource),
                        ReplayableDataStreamList.replay(constSource),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(IterationConfig.OperatorLifeCycle.PER_ROUND)
                                .build(),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<Integer> reducer =
                                    variableStreams
                                            .<Integer>get(0)
                                            .connect(dataStreams.<Integer>get(0))
                                            .transform(
                                                    "Reducer",
                                                    BasicTypeInfo.INT_TYPE_INFO,
                                                    new TwoInputReducePerRoundOperator())
                                            .setParallelism(1);

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
}
