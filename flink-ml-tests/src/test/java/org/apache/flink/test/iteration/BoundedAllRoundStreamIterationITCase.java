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
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.test.iteration.operators.CollectSink;
import org.apache.flink.test.iteration.operators.EpochRecord;
import org.apache.flink.test.iteration.operators.IncrementEpochMap;
import org.apache.flink.test.iteration.operators.OutputRecord;
import org.apache.flink.test.iteration.operators.SequenceSource;
import org.apache.flink.test.iteration.operators.StatefulProcessFunction;
import org.apache.flink.test.iteration.operators.TwoInputReduceAllRoundProcessFunction;
import org.apache.flink.testutils.junit.SharedObjects;
import org.apache.flink.testutils.junit.SharedReference;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.TestLogger;

import org.apache.commons.collections.IteratorUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import javax.annotation.Nullable;

import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.computeRoundStat;
import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.createMiniClusterConfiguration;
import static org.apache.flink.test.iteration.UnboundedStreamIterationITCase.verifyResult;
import static org.junit.Assert.assertEquals;

/**
 * Tests the cases of {@link Iterations#iterateBoundedStreamsUntilTermination(DataStreamList,
 * ReplayableDataStreamList, IterationConfig, IterationBody)} that using all-round iterations.
 */
public class BoundedAllRoundStreamIterationITCase extends TestLogger {

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

    @Test(timeout = 60000)
    public void testSyncVariableOnlyBoundedIteration() throws Exception {
        JobGraph jobGraph =
                createVariableOnlyJobGraph(4, 1000, false, 0, true, 4, null, false, result);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(6, result.get().size());
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(result.get(), OutputRecord.Event.EPOCH_WATERMARK_INCREMENTED, 5);

        verifyResult(roundsStat, 5, 1, 4 * (0 + 999) * 1000 / 2);
        assertEquals(OutputRecord.Event.TERMINATED, result.get().take().getEvent());
    }

    @Test
    public void testSyncVariableOnlyBoundedIterationWithVariableTerminationCriteria()
            throws Exception {
        testSyncVariableOnlyBoundedIterationWithTerminationCriteria(false);
    }

    @Test
    public void testSyncVariableOnlyBoundedIterationWithConstantTerminationCriteria()
            throws Exception {
        testSyncVariableOnlyBoundedIterationWithTerminationCriteria(true);
    }

    private void testSyncVariableOnlyBoundedIterationWithTerminationCriteria(
            boolean terminationCriteriaFollowsConstantsStreams) throws Exception {
        JobGraph jobGraph =
                createVariableOnlyJobGraph(
                        4,
                        1000,
                        false,
                        0,
                        true,
                        40,
                        4,
                        terminationCriteriaFollowsConstantsStreams,
                        result);
        miniCluster.executeJobBlocking(jobGraph);

        // If termination criteria is created only with the constants streams, it would not have
        // records after the round 1 if the input is not replayed.
        int numOfRound = terminationCriteriaFollowsConstantsStreams ? 1 : 4;
        assertEquals(numOfRound + 1, result.get().size());

        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(
                        result.get(), OutputRecord.Event.EPOCH_WATERMARK_INCREMENTED, numOfRound);

        verifyResult(roundsStat, numOfRound, 1, 4 * (0 + 999) * 1000 / 2);
        assertEquals(OutputRecord.Event.TERMINATED, result.get().take().getEvent());
    }

    @Test(timeout = 60000)
    public void testSyncVariableAndConstantBoundedIteration() throws Exception {
        JobGraph jobGraph = createVariableAndConstantJobGraph(4, 1000, false, 0, true, 4, result);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(6, result.get().size());
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(result.get(), OutputRecord.Event.EPOCH_WATERMARK_INCREMENTED, 5);

        verifyResult(roundsStat, 5, 1, 4 * (0 + 999) * 1000 / 2);
        assertEquals(OutputRecord.Event.TERMINATED, result.get().take().getEvent());
    }

    @Test
    public void testBoundedIterationWithEndInput() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        env.getConfig().enableObjectReuse();

        DataStream<Integer> inputStream = env.fromElements(1, 2, 3);

        DataStreamList outputs =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(inputStream),
                        ReplayableDataStreamList.replay(inputStream),
                        IterationConfig.newBuilder().build(),
                        (variableStreams, dataStreams) -> {
                            DataStream<Integer> variables = variableStreams.get(0);
                            DataStream<Integer> result =
                                    dataStreams
                                            .<Integer>get(0)
                                            .transform(
                                                    "sum",
                                                    BasicTypeInfo.INT_TYPE_INFO,
                                                    new SumOperator());
                            return new IterationBodyResult(
                                    DataStreamList.of(variables),
                                    DataStreamList.of(result),
                                    variables.flatMap(new TerminateOnMaxIter<>(10)));
                        });
        List<Integer> result = IteratorUtils.toList(outputs.get(0).executeAndCollect());
        result.forEach(r -> r.equals(60));
    }

    private static JobGraph createVariableOnlyJobGraph(
            int numSources,
            int numRecordsPerSource,
            boolean holdSource,
            int period,
            boolean sync,
            int maxRound,
            @Nullable Integer terminationCriteriaRound,
            boolean terminationCriteriaFollowsConstantsStreams,
            SharedReference<BlockingQueue<OutputRecord<Integer>>> result) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(1);
        env.setMaxParallelism(5);
        DataStream<EpochRecord> variableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<EpochRecord>() {})
                        .setParallelism(numSources)
                        .name("Variable");
        DataStream<EpochRecord> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, holdSource, period))
                        .setParallelism(numSources)
                        .name("Constant");
        DataStreamList outputs =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(variableSource),
                        ReplayableDataStreamList.notReplay(constSource),
                        IterationConfig.newBuilder().build(),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<EpochRecord> reducer =
                                    variableStreams
                                            .<EpochRecord>get(0)
                                            .connect(dataStreams.<EpochRecord>get(0))
                                            .process(
                                                    new TwoInputReduceAllRoundProcessFunction(
                                                            sync, maxRound));

                            return new IterationBodyResult(
                                    DataStreamList.of(
                                            reducer.partitionCustom(
                                                            (k, numPartitions) -> k % numPartitions,
                                                            EpochRecord::getValue)
                                                    .map(x -> x)
                                                    .keyBy(EpochRecord::getValue)
                                                    .process(
                                                            new StatefulProcessFunction<
                                                                    EpochRecord>() {})
                                                    .setParallelism(4)
                                                    .map(new IncrementEpochMap())
                                                    .setParallelism(numSources)),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    new OutputTag<OutputRecord<Integer>>(
                                                            "output") {})),
                                    terminationCriteriaRound == null
                                            ? null
                                            : (terminationCriteriaFollowsConstantsStreams
                                                            ? dataStreams.<EpochRecord>get(0)
                                                            : reducer)
                                                    .flatMap(
                                                            new TerminateOnMaxIter(
                                                                    terminationCriteriaRound)));
                        });
        outputs.<OutputRecord<Integer>>get(0).addSink(new CollectSink(result));

        return env.getStreamGraph().getJobGraph();
    }

    private static JobGraph createVariableAndConstantJobGraph(
            int numSources,
            int numRecordsPerSource,
            boolean holdSource,
            int period,
            boolean sync,
            int maxRound,
            SharedReference<BlockingQueue<OutputRecord<Integer>>> result) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(1);
        env.setMaxParallelism(5);
        DataStream<EpochRecord> variableSource =
                env.addSource(new DraftExecutionEnvironment.EmptySource<EpochRecord>() {})
                        .setParallelism(numSources)
                        .name("Variable");
        DataStream<EpochRecord> constSource =
                env.addSource(new SequenceSource(numRecordsPerSource, holdSource, period))
                        .setParallelism(numSources)
                        .name("Constant");
        DataStreamList outputs =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(variableSource),
                        ReplayableDataStreamList.notReplay(constSource),
                        IterationConfig.newBuilder().build(),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator<EpochRecord> reducer =
                                    variableStreams
                                            .<EpochRecord>get(0)
                                            .connect(dataStreams.<EpochRecord>get(0))
                                            .process(
                                                    new TwoInputReduceAllRoundProcessFunction(
                                                            sync, maxRound));

                            SingleOutputStreamOperator<EpochRecord> feedbackStream =
                                    reducer.partitionCustom(
                                                    (k, numPartitions) -> k % numPartitions,
                                                    EpochRecord::getValue)
                                            .map(x -> x)
                                            .keyBy(EpochRecord::getValue)
                                            .process(new StatefulProcessFunction<EpochRecord>() {})
                                            .setParallelism(4)
                                            .map(new IncrementEpochMap())
                                            .setParallelism(numSources);

                            return new IterationBodyResult(
                                    DataStreamList.of(feedbackStream),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    new OutputTag<OutputRecord<Integer>>(
                                                            "output") {})));
                        });
        outputs.<OutputRecord<Integer>>get(0).addSink(new CollectSink(result));

        return env.getStreamGraph().getJobGraph();
    }

    private static class SumOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Integer, Integer>, BoundedOneInput {

        private int sum = 0;

        @Override
        public void processElement(StreamRecord<Integer> element) {
            sum += element.getValue();
        }

        @Override
        public void endInput() {
            output.collect(new StreamRecord<>(sum));
        }

        @Override
        public void finish() {
            output.collect(new StreamRecord<>(sum));
        }

        @Override
        public void close() {
            output.collect(new StreamRecord<>(sum));
        }
    }
}
