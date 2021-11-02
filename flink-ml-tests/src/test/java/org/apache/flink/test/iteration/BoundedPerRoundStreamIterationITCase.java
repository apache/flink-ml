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
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.iteration.operators.CollectSink;
import org.apache.flink.test.iteration.operators.EpochRecord;
import org.apache.flink.test.iteration.operators.OutputRecord;
import org.apache.flink.test.iteration.operators.SequenceSource;
import org.apache.flink.test.iteration.operators.StatefulProcessFunction;
import org.apache.flink.test.iteration.operators.TwoInputReducePerRoundOperator;
import org.apache.flink.testutils.junit.SharedObjects;
import org.apache.flink.testutils.junit.SharedReference;
import org.apache.flink.util.TestLogger;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.computeRoundStat;
import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.createMiniClusterConfiguration;
import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.verifyResult;
import static org.junit.Assert.assertEquals;

/** Tests the per-round iterations. */
public class BoundedPerRoundStreamIterationITCase extends TestLogger {

    @Rule public final SharedObjects sharedObjects = SharedObjects.create();

    private MiniCluster miniCluster;

    private SharedReference<BlockingQueue<OutputRecord<Integer>>> result;

    @Before
    public void setup() throws Exception {
        miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2));
        miniCluster.start();

        result = sharedObjects.add(new LinkedBlockingQueue<>());
    }

    @After
    public void teardown() throws Exception {
        if (miniCluster != null) {
            miniCluster.close();
        }
    }

    @Test
    public void testPerRoundIteration() throws Exception {
        JobGraph jobGraph = createPerRoundJobGraph(4, 1000, 5, result);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(5, result.get().size());
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(result.get(), OutputRecord.Event.TERMINATED, 5);
        verifyResult(roundsStat, 5, 1, 4 * (0 + 999) * 1000 / 2);
    }

    private static JobGraph createPerRoundJobGraph(
            int numSources,
            int numRecordsPerSource,
            int maxRound,
            SharedReference<BlockingQueue<OutputRecord<Integer>>> result) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataStream<Integer> variableSource = env.fromElements(0);
        DataStream<Integer> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, false, 0))
                        .setParallelism(numSources)
                        .map(EpochRecord::getValue)
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
                                    DataStreamList.of(
                                            reducer.keyBy(x -> x)
                                                    .process(
                                                            new StatefulProcessFunction<
                                                                    Integer>() {})
                                                    .setParallelism(4)
                                                    .filter(x -> x < maxRound)
                                                    .setParallelism(1)),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    TwoInputReducePerRoundOperator.OUTPUT_TAG)),
                                    reducer.filter(x -> x < maxRound).setParallelism(1));
                        });
        outputs.<OutputRecord<Integer>>get(0).addSink(new CollectSink(result));

        return env.getStreamGraph().getJobGraph();
    }
}
