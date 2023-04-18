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

import org.apache.flink.api.common.functions.JoinFunction;
import org.apache.flink.api.common.state.BroadcastState;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationConfig.OperatorLifeCycle;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
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

import java.util.ArrayList;
import java.util.List;
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

    private SharedReference<BlockingQueue<OutputRecord<Integer>>> collectedOutputRecord;
    private SharedReference<BlockingQueue<Long>> collectedWatermarks;
    private SharedReference<BlockingQueue<Long>> collectedOutputs;

    @Before
    public void setup() throws Exception {
        miniCluster = new MiniCluster(createMiniClusterConfiguration(2, 2));
        miniCluster.start();

        collectedOutputRecord = sharedObjects.add(new LinkedBlockingQueue<>());
        collectedWatermarks = sharedObjects.add(new LinkedBlockingQueue<>());
        collectedOutputs = sharedObjects.add(new LinkedBlockingQueue<>());
    }

    @After
    public void teardown() throws Exception {
        if (miniCluster != null) {
            miniCluster.close();
        }
    }

    @Test
    public void testPerRoundIteration() throws Exception {
        JobGraph jobGraph = createPerRoundJobGraph(4, 1000, 5, collectedOutputRecord);
        miniCluster.executeJobBlocking(jobGraph);

        assertEquals(5, collectedOutputRecord.get().size());
        Map<Integer, Tuple2<Integer, Integer>> roundsStat =
                computeRoundStat(collectedOutputRecord.get(), OutputRecord.Event.TERMINATED, 5);
        verifyResult(roundsStat, 5, 1, 4 * (0 + 999) * 1000 / 2);
    }

    @Test
    public void testPerRoundIterationWithJoin() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataStream<Tuple2<Long, Integer>> input1 = env.fromElements(Tuple2.of(1L, 1));

        DataStream<Tuple2<Long, Long>> input2 = env.fromElements(Tuple2.of(1L, 2L));

        DataStream<Tuple2<Long, Long>> iterationWithJoinResult =
                Iterations.iterateBoundedStreamsUntilTermination(
                                DataStreamList.of(input1),
                                ReplayableDataStreamList.replay(input2),
                                IterationConfig.newBuilder()
                                        .setOperatorLifeCycle(
                                                IterationConfig.OperatorLifeCycle.PER_ROUND)
                                        .build(),
                                new IterationBodyWithJoin())
                        .get(0);
        DataStream<Long> watermarks =
                iterationWithJoinResult.transform(
                        "CollectingWatermark", Types.LONG, new CollectingWatermark());

        watermarks.addSink(new LongSink(collectedWatermarks));

        JobGraph graph = env.getStreamGraph().getJobGraph();
        miniCluster.executeJobBlocking(graph);

        assertEquals(env.getParallelism(), collectedWatermarks.get().size());
        collectedWatermarks
                .get()
                .iterator()
                .forEachRemaining(x -> assertEquals(Long.MAX_VALUE, (long) x));
    }

    @Test
    public void testPerRoundIterationWithState() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        DataStream<Long> broadcastStream = env.fromElements(1L);
        DataStream<Long> inputStream = env.fromElements(1L);
        DataStreamList outputStream =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(inputStream),
                        ReplayableDataStreamList.replay(broadcastStream),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(OperatorLifeCycle.PER_ROUND)
                                .build(),
                        new PerRoundIterationBodyWithState());

        outputStream.<Long>get(0).addSink(new LongSink(collectedOutputs));
        JobGraph jobGraph = env.getStreamGraph().getJobGraph();
        miniCluster.executeJobBlocking(jobGraph);

        List<Long> result = new ArrayList<>(3);
        collectedOutputs.get().drainTo(result);
        assertEquals(3, result.size());
        for (long value : result) {
            assertEquals(1L, value);
        }
    }

    private static JobGraph createPerRoundJobGraph(
            int numSources,
            int numRecordsPerSource,
            int maxRound,
            SharedReference<BlockingQueue<OutputRecord<Integer>>> result) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.getConfig().enableObjectReuse();
        env.setParallelism(1);
        env.setMaxParallelism(5);

        DataStream<Integer> variableSource = env.fromElements(0);
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
                                            reducer.partitionCustom(
                                                            (k, numPartitions) -> k % numPartitions,
                                                            x -> x)
                                                    .map(x -> x)
                                                    .keyBy(x -> x)
                                                    .process(
                                                            new StatefulProcessFunction<
                                                                    Integer>() {})
                                                    .setParallelism(4)
                                                    .filter(x -> x < maxRound)
                                                    .setParallelism(1)),
                                    DataStreamList.of(
                                            reducer.getSideOutput(
                                                    TwoInputReducePerRoundOperator.OUTPUT_TAG)),
                                    reducer.filter(x -> x < maxRound)
                                            .map(x -> (double) x)
                                            .setParallelism(1));
                        });
        outputs.<OutputRecord<Integer>>get(0).addSink(new CollectSink(result));

        return env.getStreamGraph().getJobGraph();
    }

    private static class IterationBodyWithJoin implements IterationBody {
        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<Tuple2<Long, Integer>> input1 = variableStreams.get(0);
            DataStream<Tuple2<Long, Long>> input2 = dataStreams.get(0);

            DataStream<Long> terminationCriteria = input1.flatMap(new TerminateOnMaxIter(1));

            DataStream<Tuple2<Long, Long>> res =
                    input1.join(input2)
                            .where(x -> x.f0)
                            .equalTo(x -> x.f0)
                            .window(EndOfStreamWindows.get())
                            .apply(
                                    new JoinFunction<
                                            Tuple2<Long, Integer>,
                                            Tuple2<Long, Long>,
                                            Tuple2<Long, Long>>() {
                                        @Override
                                        public Tuple2<Long, Long> join(
                                                Tuple2<Long, Integer> longIntegerTuple2,
                                                Tuple2<Long, Long> longLongTuple2) {
                                            return longLongTuple2;
                                        }
                                    });

            return new IterationBodyResult(
                    DataStreamList.of(input1), DataStreamList.of(res), terminationCriteria);
        }
    }

    private static class PerRoundIterationBodyWithState implements IterationBody {

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<Long> variableStream = variableStreams.get(0);

            DataStream<Long> feedback =
                    variableStream.transform("mapWithState", Types.LONG, new MapWithState());

            DataStream<Integer> terminationCriteria =
                    feedback.<Long>flatMap(new TerminateOnMaxIter(2)).returns(Types.INT);

            return new IterationBodyResult(
                    DataStreamList.of(feedback), DataStreamList.of(feedback), terminationCriteria);
        }
    }

    private static class MapWithState extends AbstractStreamOperator<Long>
            implements OneInputStreamOperator<Long, Long> {
        private ListState<Long> listState;
        private ListState<Long> unionState;
        private BroadcastState<Long, Long> broadcastState;

        @Override
        public void processElement(StreamRecord<Long> element) throws Exception {
            long val = element.getValue();
            listState.add(val);
            unionState.add(val);
            broadcastState.put(val, val);
            output.collect(element);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            listState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("longState", Types.LONG));
            unionState =
                    context.getOperatorStateStore()
                            .getUnionListState(new ListStateDescriptor<>("unionState", Types.LONG));
            broadcastState =
                    context.getOperatorStateStore()
                            .getBroadcastState(
                                    new MapStateDescriptor<>(
                                            "broadcastState", Types.LONG, Types.LONG));
        }
    }

    private static class LongSink implements SinkFunction<Long> {
        private final SharedReference<BlockingQueue<Long>> collectedLong;

        public LongSink(SharedReference<BlockingQueue<Long>> collectedLong) {
            this.collectedLong = collectedLong;
        }

        @Override
        public void invoke(Long value, Context context) {
            collectedLong.get().add(value);
        }
    }

    private static class CollectingWatermark extends AbstractStreamOperator<Long>
            implements OneInputStreamOperator<Tuple2<Long, Long>, Long> {

        @Override
        public void processElement(StreamRecord<Tuple2<Long, Long>> streamRecord) {}

        @Override
        public void processWatermark(Watermark mark) throws Exception {
            super.processWatermark(mark);
            output.collect(new StreamRecord<>(mark.getTimestamp()));
        }
    }
}
