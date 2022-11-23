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
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.iteration.operators.CollectSink;
import org.apache.flink.test.iteration.operators.EpochRecord;
import org.apache.flink.test.iteration.operators.IncrementEpochMap;
import org.apache.flink.test.iteration.operators.OutputRecord;
import org.apache.flink.test.iteration.operators.SequenceSource;
import org.apache.flink.test.iteration.operators.StatefulProcessFunction;
import org.apache.flink.test.iteration.operators.TwoInputReduceAllRoundProcessFunction;
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

/** Tests the iterations with mixed operator life-cycles. */
public class BoundedMixedLifeCycleStreamIterationITCase extends TestLogger {

    @Rule public final SharedObjects sharedObjects = SharedObjects.create();

    private MiniCluster miniCluster;

    private SharedReference<BlockingQueue<OutputRecord<Integer>>> allRoundResult;

    private SharedReference<BlockingQueue<OutputRecord<Integer>>> perRoundResult;

    @Before
    public void setup() throws Exception {
        miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2));
        miniCluster.start();

        allRoundResult = sharedObjects.add(new LinkedBlockingQueue<>());
        perRoundResult = sharedObjects.add(new LinkedBlockingQueue<>());
    }

    @After
    public void teardown() throws Exception {
        if (miniCluster != null) {
            miniCluster.close();
        }
    }

    @Test
    public void testIterationBodyWithMixedLifeCycle() throws Exception {
        JobGraph jobGraph =
                createJobGraphWithMixedLifeCycle(4, 1000, 5, allRoundResult, perRoundResult);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(6, perRoundResult.get().size());
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(perRoundResult.get(), OutputRecord.Event.TERMINATED, 6);
        verifyResult(roundsStat, 6, 1, 4 * (0 + 999) * 1000 / 2);

        assertEquals(7, allRoundResult.get().size());
        roundsStat =
                computeRoundStat(
                        allRoundResult.get(), OutputRecord.Event.EPOCH_WATERMARK_INCREMENTED, 6);
        verifyResult(roundsStat, 6, 1, 4 * (0 + 999) * 1000 / 2);
        assertEquals(OutputRecord.Event.TERMINATED, allRoundResult.get().take().getEvent());
    }

    private static JobGraph createJobGraphWithMixedLifeCycle(
            int numSources,
            int numRecordsPerSource,
            int maxRound,
            SharedReference<BlockingQueue<OutputRecord<Integer>>> allRoundResult,
            SharedReference<BlockingQueue<OutputRecord<Integer>>> perRoundResult) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(1);

        DataStream<EpochRecord> allRoundVariableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<EpochRecord>() {});
        DataStream<Integer> perRoundVariableSource = env.fromElements(0);

        DataStream<EpochRecord> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, false, 0))
                        .setParallelism(numSources)
                        .name("Constants");

        DataStreamList outputs =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(perRoundVariableSource, allRoundVariableSource),
                        ReplayableDataStreamList.replay(constSource.map(x -> x))
                                .andNotReplay(constSource.map(x -> x)),
                        IterationConfig.newBuilder().build(),
                        new MixedLifeCycleIterationBody(maxRound));

        outputs.<OutputRecord<Integer>>get(0).addSink(new CollectSink(perRoundResult));
        outputs.<OutputRecord<Integer>>get(1).addSink(new CollectSink(allRoundResult));

        return env.getStreamGraph().getJobGraph();
    }

    private static class MixedLifeCycleIterationBody implements IterationBody {

        private final int maxRound;

        public MixedLifeCycleIterationBody(int maxRound) {
            this.maxRound = maxRound;
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            SingleOutputStreamOperator<Integer> perRoundReducer =
                    (SingleOutputStreamOperator)
                            IterationBody.forEachRound(
                                            DataStreamList.of(
                                                    variableStreams.get(0), dataStreams.get(0)),
                                            streams -> {
                                                DataStream<Integer> variableStream = streams.get(0);
                                                DataStream<EpochRecord> replayedDataStream =
                                                        streams.get(1);

                                                return DataStreamList.of(
                                                        variableStream
                                                                .connect(
                                                                        replayedDataStream.map(
                                                                                EpochRecord
                                                                                        ::getValue))
                                                                .transform(
                                                                        "Reducer",
                                                                        BasicTypeInfo.INT_TYPE_INFO,
                                                                        new TwoInputReducePerRoundOperator())
                                                                .setParallelism(1));
                                            })
                                    .get(0);

            SingleOutputStreamOperator<EpochRecord> allRoundReducer =
                    variableStreams
                            .<EpochRecord>get(1)
                            .connect(dataStreams.<EpochRecord>get(1))
                            .process(new TwoInputReduceAllRoundProcessFunction(true, maxRound));

            return new IterationBodyResult(
                    DataStreamList.of(
                            perRoundReducer
                                    .keyBy(x -> x)
                                    .process(new StatefulProcessFunction<Integer>() {})
                                    .setParallelism(4)
                                    .filter(x -> x <= maxRound)
                                    .setParallelism(1),
                            allRoundReducer.map(new IncrementEpochMap())),
                    DataStreamList.of(
                            perRoundReducer.getSideOutput(
                                    TwoInputReducePerRoundOperator.OUTPUT_TAG),
                            allRoundReducer.getSideOutput(
                                    TwoInputReducePerRoundOperator.OUTPUT_TAG)));
        }
    }
}
