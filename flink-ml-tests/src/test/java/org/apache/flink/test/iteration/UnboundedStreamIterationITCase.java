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

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.test.iteration.operators.CollectSink;
import org.apache.flink.test.iteration.operators.EpochRecord;
import org.apache.flink.test.iteration.operators.IncrementEpochMap;
import org.apache.flink.test.iteration.operators.OutputRecord;
import org.apache.flink.test.iteration.operators.ReduceAllRoundProcessFunction;
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

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import static org.junit.Assert.assertEquals;

/** Integration cases for unbounded iteration. */
public class UnboundedStreamIterationITCase extends TestLogger {

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
    public void testVariableOnlyUnboundedIteration() throws Exception {
        // Create the test job
        JobGraph jobGraph = createVariableOnlyJobGraph(4, 1000, true, 0, false, 1, false, result);
        miniCluster.submitJob(jobGraph);

        // Expected records is round * parallelism * numRecordsPerSource
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(result.get(), OutputRecord.Event.PROCESS_ELEMENT, 2 * 4 * 1000);
        verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
    }

    @Test(timeout = 60000)
    public void testVariableOnlyBoundedIteration() throws Exception {
        // Create the test job
        JobGraph jobGraph = createVariableOnlyJobGraph(4, 1000, false, 0, false, 1, false, result);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(8001, result.get().size());

        // Expected records is round * parallelism * numRecordsPerSource
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(result.get(), OutputRecord.Event.PROCESS_ELEMENT, 2 * 4 * 1000);
        verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
        assertEquals(OutputRecord.Event.TERMINATED, result.get().take().getEvent());
    }

    @Test(timeout = 60000)
    public void testVariableOnlyBoundedIterationWithBroadcast() throws Exception {
        // Create the test job
        JobGraph jobGraph = createVariableOnlyJobGraph(4, 1000, false, 0, false, 1, true, result);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(8001, result.get().size());

        // Expected records is round * parallelism * numRecordsPerSource * parallelism of reduce
        // operators
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(
                        result.get(), OutputRecord.Event.PROCESS_ELEMENT, 2 * 4 * 1000 * 1);
        verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
        assertEquals(OutputRecord.Event.TERMINATED, result.get().take().getEvent());
    }

    @Test(timeout = 60000)
    public void testVariableAndConstantsUnboundedIteration() throws Exception {
        // Create the test job
        JobGraph jobGraph = createVariableAndConstantJobGraph(4, 1000, true, 0, false, 1, result);
        miniCluster.submitJob(jobGraph);

        // Expected records is round * parallelism * numRecordsPerSource
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(result.get(), OutputRecord.Event.PROCESS_ELEMENT, 2 * 4 * 1000);
        verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
    }

    @Test(timeout = 60000)
    public void testVariableAndConstantBoundedIteration() throws Exception {
        // Create the test job
        JobGraph jobGraph = createVariableAndConstantJobGraph(4, 1000, false, 0, false, 1, result);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(8001, result.get().size());

        // Expected records is round * parallelism * numRecordsPerSource
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(result.get(), OutputRecord.Event.PROCESS_ELEMENT, 2 * 4 * 1000);
        verifyResult(roundsStat, 2, 4000, 4 * (0 + 999) * 1000 / 2);
        assertEquals(OutputRecord.Event.TERMINATED, result.get().take().getEvent());
    }

    @Test
    public void testBoundedIterationWithSideOutput() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        env.getConfig().enableObjectReuse();

        final OutputTag<Integer> outputTag = new OutputTag("0", Types.INT) {};
        final Integer[] sourceData = new Integer[] {1, 2, 3};

        DataStream<Integer> variableStream =
                env.addSource(new DraftExecutionEnvironment.EmptySource<Integer>() {});
        DataStream<Integer> dataStream = env.fromElements(sourceData);

        DataStreamList result =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(variableStream),
                        DataStreamList.of(dataStream),
                        (variableStreams, dataStreams) -> {
                            SingleOutputStreamOperator transformed =
                                    dataStreams
                                            .<Integer>get(0)
                                            .transform(
                                                    "side-output",
                                                    Types.INT,
                                                    new SideOutputOperator(outputTag));
                            return new IterationBodyResult(
                                    DataStreamList.of(variableStreams.get(0)),
                                    DataStreamList.of(transformed.getSideOutput(outputTag)));
                        });
        assertEquals(
                Arrays.asList(sourceData), IteratorUtils.toList(result.get(0).executeAndCollect()));
    }

    public static MiniClusterConfiguration createMiniClusterConfiguration(int numTm, int numSlot) {
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.BIND_PORT, "18081-19091");
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        return new MiniClusterConfiguration.Builder()
                .setConfiguration(configuration)
                .setNumTaskManagers(numTm)
                .setNumSlotsPerTaskManager(numSlot)
                .build();
    }

    private static JobGraph createVariableOnlyJobGraph(
            int numSources,
            int numRecordsPerSource,
            boolean holdSource,
            int period,
            boolean sync,
            int maxRound,
            boolean doBroadcast,
            SharedReference<BlockingQueue<OutputRecord<Integer>>> result) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(1);
        DataStream<EpochRecord> source =
                env.addSource(new SequenceSource(numRecordsPerSource, holdSource, period))
                        .setParallelism(numSources);
        DataStreamList outputs =
                Iterations.iterateUnboundedStreams(
                        DataStreamList.of(source),
                        DataStreamList.of(),
                        (variableStreams, dataStreams) -> {
                            DataStream<EpochRecord> variable = variableStreams.get(0);
                            if (doBroadcast) {
                                variable = variable.broadcast();
                            }

                            SingleOutputStreamOperator<EpochRecord> reducer =
                                    variable.process(
                                            new ReduceAllRoundProcessFunction(sync, maxRound));
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
                                                            "output") {})));
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

    private static class SideOutputOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Integer, Integer> {

        private final OutputTag<Integer> outputTag;

        public SideOutputOperator(OutputTag<Integer> outputTag) {
            this.outputTag = outputTag;
        }

        @Override
        public void open() throws Exception {
            super.open();
            StreamConfig config = getOperatorConfig();
            ClassLoader cl = getClass().getClassLoader();

            assertEquals(IntSerializer.INSTANCE, config.getTypeSerializerIn(0, cl));
            assertEquals(IntSerializer.INSTANCE, config.getTypeSerializerOut(cl));
            assertEquals(IntSerializer.INSTANCE, config.getTypeSerializerSideOut(outputTag, cl));
        }

        @Override
        public void processElement(StreamRecord<Integer> element) {
            output.collect(outputTag, element);
        }
    }

    static Map<Integer, Tuple2<Integer, Integer>> computeRoundStat(
            BlockingQueue<OutputRecord<Integer>> result,
            OutputRecord.Event event,
            int expectedRecords)
            throws InterruptedException {
        Map<Integer, Tuple2<Integer, Integer>> roundsStat = new HashMap<>();
        for (int i = 0; i < expectedRecords; ++i) {
            OutputRecord<Integer> next = result.take();
            assertEquals(event, next.getEvent());
            Tuple2<Integer, Integer> state =
                    roundsStat.computeIfAbsent(next.getRound(), ignored -> new Tuple2<>(0, 0));
            state.f0++;
            state.f1 = next.getValue();
        }

        return roundsStat;
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
