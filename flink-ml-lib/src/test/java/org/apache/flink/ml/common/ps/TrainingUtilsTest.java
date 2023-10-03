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

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.iterations.AllReduceStage;
import org.apache.flink.ml.common.ps.iterations.IterationStageList;
import org.apache.flink.ml.common.ps.iterations.MLSessionImpl;
import org.apache.flink.ml.common.ps.iterations.ProcessStage;
import org.apache.flink.ml.common.ps.iterations.PullStage;
import org.apache.flink.ml.common.ps.iterations.PushStage;
import org.apache.flink.ml.common.ps.iterations.ReduceScatterStage;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.ml.common.ps.typeinfo.Long2ObjectOpenHashMapTypeInfo;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.ml.common.ps.utils.ProxySideOutput;
import org.apache.flink.ml.common.ps.utils.TrainingUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableSupplier;

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link TrainingUtils}. */
public class TrainingUtilsTest {
    private static final int NUM_WORKERS = 2;
    private static final int NUM_SERVERS = 6;
    private static final int MAX_ITER = 3;
    private static final int NUM_DOUBLES_PER_KEY = 2;
    private DataStream<DenseVector> inputData;
    StreamExecutionEnvironment env;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        env.setParallelism(NUM_WORKERS);
        inputData =
                env.fromCollection(
                                Arrays.asList(
                                        Vectors.dense(1, 1, 1, 1),
                                        Vectors.dense(2, 2, 2, 2),
                                        Vectors.dense(3, 3, 3, 3),
                                        Vectors.dense(4, 4, 4, 4)))
                        .map(x -> x, DenseVectorTypeInfo.INSTANCE);
    }

    @Test
    public void testPushSumAndPullAgg() throws Exception {
        MockSession mockSession = new MockSession();

        IterationStageList<MockSession> stageList =
                new IterationStageList<>(mockSession)
                        .addStage(
                                new PushStage(
                                        () -> new SharedLongArray(new long[] {1, 4}),
                                        () -> new SharedDoubleArray(new double[] {1, 1, 4, 4})))
                        .addStage(
                                new PullStage(
                                        () -> new SharedLongArray(new long[] {1, 3, 4}),
                                        () -> {
                                            mockSession.pullResult.size(4);
                                            return mockSession.pullResult;
                                        },
                                        new MockAggregator()))
                        .addStage(
                                new ResultChecker(
                                        () -> {
                                            double[] expectedResult = new double[4];
                                            Arrays.fill(
                                                    expectedResult,
                                                    (mockSession.iterationId + 1)
                                                            * (mockSession.iterationId + 1)
                                                            * 68);
                                            return Arrays.equals(
                                                    expectedResult,
                                                    trimToArray(mockSession.pullResult));
                                        }))
                        .setTerminationCriteria(session -> session.iterationId >= MAX_ITER);

        DataStreamList resultList =
                TrainingUtils.train(
                        inputData,
                        stageList,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        new MockModelUpdater(NUM_DOUBLES_PER_KEY),
                        NUM_SERVERS);

        DataStream<Tuple2<Long, double[]>> modelStream = resultList.get(0);
        List<Tuple2<Long, double[]>> collectedModelData =
                IteratorUtils.toList(modelStream.executeAndCollect());
        List<Tuple2<Long, double[]>> expectedModelData =
                Arrays.asList(
                        Tuple2.of(
                                1L, new double[] {NUM_WORKERS * MAX_ITER, NUM_WORKERS * MAX_ITER}),
                        Tuple2.of(3L, new double[] {0, 0}),
                        Tuple2.of(
                                4L,
                                new double[] {
                                    NUM_WORKERS * MAX_ITER * 4, NUM_WORKERS * MAX_ITER * 4
                                }));

        verifyModelData(expectedModelData, collectedModelData);
    }

    @Test
    public void testPushMinAndPull() throws Exception {
        MockSession mockSession = new MockSession();

        IterationStageList<MockSession> stageList =
                new IterationStageList<>(mockSession)
                        .addStage(
                                new PushStage(
                                        () -> new SharedLongArray(new long[] {1, 4}),
                                        () -> new SharedDoubleArray(new double[] {1, 1, 4, 4}),
                                        Double::min))
                        .addStage(
                                new PullStage(
                                        () -> new SharedLongArray(new long[] {1, 3}),
                                        () -> {
                                            mockSession.pullResult.size(4);
                                            return mockSession.pullResult;
                                        }))
                        .addStage(
                                new ResultChecker(
                                        () ->
                                                Arrays.equals(
                                                        new double[] {
                                                            mockSession.iterationId + 1,
                                                            mockSession.iterationId + 1,
                                                            0,
                                                            0
                                                        },
                                                        trimToArray(mockSession.pullResult))))
                        .setTerminationCriteria(session -> session.iterationId >= MAX_ITER);

        DataStreamList resultList =
                TrainingUtils.train(
                        inputData,
                        stageList,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        new MockModelUpdater(NUM_DOUBLES_PER_KEY),
                        NUM_SERVERS);
        DataStream<Tuple3<Long, Long, double[]>> modelStream = resultList.get(0);
        List<Tuple2<Long, double[]>> collectedModelData =
                IteratorUtils.toList(modelStream.executeAndCollect());
        List<Tuple2<Long, double[]>> expectedModelData =
                Arrays.asList(
                        Tuple2.of(1L, new double[] {MAX_ITER, MAX_ITER}),
                        Tuple2.of(3L, new double[] {0, 0}),
                        Tuple2.of(4L, new double[] {MAX_ITER * 4, MAX_ITER * 4}));

        verifyModelData(expectedModelData, collectedModelData);
    }

    @Test
    public void testAllReduce() throws Exception {
        ExecutionConfig executionConfig = inputData.getExecutionEnvironment().getConfig();
        int executionInterval = 2;
        TypeSerializer<MockPojo> mockPojoTypeSerializer =
                Types.POJO(MockPojo.class).createSerializer(executionConfig);
        MockSession mockSession = new MockSession();

        IterationStageList<MockSession> stageList =
                new IterationStageList<>(mockSession)
                        .addStage(new MockInitStage())
                        .addStage(
                                new AllReduceStage<>(
                                        () -> mockSession.allReduceInputAndResult,
                                        () -> mockSession.allReduceInputAndResult,
                                        (ReduceFunction<MockPojo[]>) TrainingUtilsTest::sumPojo,
                                        mockPojoTypeSerializer,
                                        executionInterval))
                        .addStage(
                                new ResultChecker(
                                        () -> {
                                            if (mockSession.iterationId % executionInterval == 0) {
                                                MockPojo[] reduceResult =
                                                        mockSession.allReduceInputAndResult;
                                                Assert.assertEquals(2, reduceResult.length);
                                                MockPojo expectedPojo =
                                                        new MockPojo(
                                                                NUM_WORKERS
                                                                        * (mockSession.iterationId
                                                                                        / executionInterval
                                                                                + 1),
                                                                NUM_WORKERS
                                                                        * (mockSession.iterationId
                                                                                        / executionInterval
                                                                                + 1)
                                                                        * 2);
                                                Assert.assertEquals(expectedPojo, reduceResult[0]);
                                                Assert.assertEquals(expectedPojo, reduceResult[1]);
                                            }
                                            return true;
                                        }))
                        .setTerminationCriteria(session -> session.iterationId >= MAX_ITER);

        DataStreamList resultList =
                TrainingUtils.train(
                        inputData,
                        stageList,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        new MockModelUpdater(NUM_DOUBLES_PER_KEY),
                        NUM_SERVERS);
        DataStream<Tuple2<Long, double[]>> modelStream = resultList.get(0);
        List<Tuple2<Long, double[]>> modelData =
                IteratorUtils.toList(modelStream.executeAndCollect());
        Assert.assertEquals(0, modelData.size());
    }

    @Test
    public void testReduceScatter() throws Exception {
        ExecutionConfig executionConfig = inputData.getExecutionEnvironment().getConfig();
        int executionInterval = 2;
        TypeSerializer<MockPojo> mockPojoTypeSerializer =
                Types.POJO(MockPojo.class).createSerializer(executionConfig);
        MockSession mockSession =
                new MockSession(
                        Collections.singletonList(
                                new OutputTag<>(
                                        "reduceScatter",
                                        new TupleTypeInfo<>(
                                                Types.INT,
                                                Types.INT,
                                                Types.OBJECT_ARRAY(Types.POJO(MockPojo.class))))));

        IterationStageList<MockSession> stageList =
                new IterationStageList<>(mockSession)
                        .addStage(new MockInitStage())
                        .addStage(
                                new ReduceScatterStage<>(
                                        () -> mockSession.reduceScatterInput,
                                        () -> mockSession.reduceScatterResult,
                                        new int[] {1, 1},
                                        (ReduceFunction<MockPojo[]>) TrainingUtilsTest::sumPojo,
                                        mockPojoTypeSerializer,
                                        executionInterval))
                        .addStage(
                                new ResultChecker(
                                        () -> {
                                            if (mockSession.iterationId % executionInterval == 0) {
                                                MockPojo[] reduceResult =
                                                        mockSession.reduceScatterResult;
                                                Assert.assertEquals(1, reduceResult.length);
                                                MockPojo expectedPojo =
                                                        new MockPojo(NUM_WORKERS, NUM_WORKERS * 2);
                                                Assert.assertEquals(expectedPojo, reduceResult[0]);
                                            }
                                            return true;
                                        }))
                        .setTerminationCriteria(session -> session.iterationId >= MAX_ITER);

        DataStreamList resultList =
                TrainingUtils.train(
                        inputData,
                        stageList,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        new MockModelUpdater(NUM_DOUBLES_PER_KEY),
                        NUM_SERVERS);
        DataStream<Tuple2<Long, double[]>> modelStream = resultList.get(0);
        List<Tuple2<Long, double[]>> modelData =
                IteratorUtils.toList(modelStream.executeAndCollect());
        Assert.assertEquals(0, modelData.size());
    }

    @Test
    public void readTrainDataAndOutput() throws Exception {
        MockSession mockSession =
                new MockSession(
                        Collections.singletonList(
                                new OutputTag<>(
                                        "numOfTrainData",
                                        new TupleTypeInfo<>(Types.INT, Types.INT, Types.INT))));

        IterationStageList<MockSession> stageList =
                new IterationStageList<>(mockSession)
                        .addStage(new ReadDataStage())
                        .addStage(
                                new AllReduceStage<>(
                                        () -> mockSession.numDataScanned,
                                        () -> mockSession.numDataScanned,
                                        TrainingUtilsTest::sumIntArray,
                                        IntSerializer.INSTANCE))
                        .addStage(new MockOutputStage<>(() -> mockSession.numDataScanned[0]))
                        .setTerminationCriteria(session -> session.iterationId >= MAX_ITER);

        DataStreamList resultList =
                TrainingUtils.train(
                        inputData,
                        stageList,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        new MockModelUpdater(NUM_DOUBLES_PER_KEY),
                        NUM_SERVERS);

        DataStream<Tuple3<Integer, Integer, Integer>> pulledStream = resultList.get(1);
        List<Tuple3<Integer, Integer, Integer>> pulls =
                IteratorUtils.toList(pulledStream.executeAndCollect());

        List<Tuple3<Integer, Integer, Integer>> expectedPulls = new ArrayList<>();
        int numDataScanned = 4;
        for (int i = 0; i < MAX_ITER; i++) {
            for (int w = 0; w < NUM_WORKERS; w++) {
                expectedPulls.add(Tuple3.of(i, w, numDataScanned));
            }
        }
        Comparator<Tuple3<Integer, Integer, Integer>> comparator =
                (o1, o2) -> {
                    int cmp = Integer.compare(o1.f0, o2.f0);
                    if (cmp == 0) {
                        cmp = Integer.compare(o1.f1, o2.f1);
                        if (cmp == 0) {
                            cmp = Integer.compare(o1.f2, o2.f2);
                        }
                    }
                    return cmp;
                };
        TestBaseUtils.compareResultCollections(expectedPulls, pulls, comparator);
    }

    /** The session that one worker can access. */
    private static class MockSession extends MLSessionImpl<DenseVector> {
        public MockPojo[] allReduceInputAndResult;
        public MockPojo[] reduceScatterInput;
        public MockPojo[] reduceScatterResult;
        public SharedDoubleArray pullResult;
        private ProxySideOutput output;
        private Integer[] numDataScanned;

        @Override
        public void setOutput(ProxySideOutput output) {
            this.output = output;
        }

        public MockSession(List<OutputTag<?>> outputTags) {
            super(outputTags);
            pullResult = new SharedDoubleArray();
            this.numDataScanned = new Integer[1];
        }

        public MockSession() {
            this(null);
        }
    }

    /** The model updater on servers. */
    private static class MockModelUpdater implements ModelUpdater<Tuple2<Long, double[]>> {
        private final int numDoublesPerKey;
        private Long2ObjectOpenHashMap<double[]> model;
        private ListState<Long2ObjectOpenHashMap<double[]>> modelDataState;

        public MockModelUpdater(int numDoublesPerKey) {
            this.numDoublesPerKey = numDoublesPerKey;
            this.model = new Long2ObjectOpenHashMap<>();
        }

        @Override
        public void update(long[] keys, double[] values) {
            Preconditions.checkState(keys.length * numDoublesPerKey == values.length);
            for (int i = 0; i < keys.length; i++) {
                long index = keys[i];
                model.putIfAbsent(index, new double[numDoublesPerKey]);
                double[] oneDimModel = model.get(index);
                for (int j = 0; j < numDoublesPerKey; j++) {
                    oneDimModel[j] += values[i * numDoublesPerKey + j];
                }
            }
        }

        @Override
        public double[] get(long[] keys) {
            double[] values = new double[keys.length * numDoublesPerKey];
            for (int i = 0; i < keys.length; i++) {
                long index = keys[i];
                model.putIfAbsent(index, new double[numDoublesPerKey]);
                double[] oneDimModel = model.get(index);
                for (int j = 0; j < numDoublesPerKey; j++) {
                    values[i * numDoublesPerKey + j] += oneDimModel[j];
                }
            }
            return values;
        }

        @Override
        public Iterator<Tuple2<Long, double[]>> getModelSegments() {
            return model.long2ObjectEntrySet().stream()
                    .map(x -> Tuple2.of(x.getLongKey(), x.getValue()))
                    .iterator();
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            modelDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "modelDataState",
                                            new Long2ObjectOpenHashMapTypeInfo<>(
                                                    PrimitiveArrayTypeInfo
                                                            .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO)));
            model =
                    OperatorStateUtils.getUniqueElement(modelDataState, "modelDataState")
                            .orElse(new Long2ObjectOpenHashMap<>());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            modelDataState.clear();
            modelDataState.add(model);
        }
    }

    /** A stage that initialize the value for all-reduce and reduce-scatter. */
    private static class MockInitStage extends ProcessStage<MockSession> {

        @Override
        public void process(MockSession session) {
            if (session.iterationId == 0) {
                session.allReduceInputAndResult = new MockPojo[2];
                session.allReduceInputAndResult[0] = new MockPojo(1, 2);
                session.allReduceInputAndResult[1] = new MockPojo(1, 2);
            }

            session.reduceScatterInput = new MockPojo[2];
            session.reduceScatterInput[0] = new MockPojo(1, 2);
            session.reduceScatterInput[1] = new MockPojo(1, 2);
            session.reduceScatterResult = new MockPojo[1];
        }
    }

    /** A stage that scans the data and count the number of data points scanned. */
    private static class ReadDataStage extends ProcessStage<MockSession> {

        @Override
        public void process(MockSession session) throws Exception {
            session.numDataScanned[0] = 0;
            while (session.inputData.hasNext()) {
                session.inputData.next();
                session.numDataScanned[0]++;
            }
            session.inputData.reset();
        }
    }

    /** A stage that checks the value of some intermediate results. */
    private static class ResultChecker extends ProcessStage<MockSession> {
        Supplier<Boolean> checker;

        public ResultChecker(SerializableSupplier<Boolean> checker) {
            this.checker = checker;
        }

        @Override
        public void process(MockSession session) {
            Preconditions.checkState(checker.get());
        }
    }

    /** A stage that output non-model data to downstream tasks. */
    private static class MockOutputStage<T> extends ProcessStage<MockSession> {

        private final SerializableSupplier<T> outputSupplier;

        public MockOutputStage(SerializableSupplier<T> outputSupplier) {
            this.outputSupplier = outputSupplier;
        }

        @Override
        public void process(MockSession session) {
            OutputTag<Tuple3<Integer, Integer, T>> outputTag =
                    (OutputTag<Tuple3<Integer, Integer, T>>) session.getOutputTags().get(0);
            session.output.output(
                    outputTag,
                    new StreamRecord<>(
                            Tuple3.of(
                                    session.iterationId, session.workerId, outputSupplier.get())));
        }
    }

    /** An aggregator that can be used in a pull request. */
    private static class MockAggregator implements PullStage.Aggregator<double[], double[]> {
        @Override
        public double[] add(double[] in, double[] acc) {
            if (acc == null) {
                acc = new double[in.length * in.length];
            }

            for (int i = 0; i < in.length; i++) {
                for (int j = 0; j < in.length; j++) {
                    acc[i * in.length + j] += in[i] * in[j];
                }
            }
            return acc;
        }

        @Override
        public double[] merge(double[] acc1, double[] acc2) {
            for (int i = 0; i < acc1.length; i++) {
                acc2[i] += acc1[i];
            }
            return acc2;
        }
    }

    private void verifyModelData(
            List<Tuple2<Long, double[]>> expected, List<Tuple2<Long, double[]>> actual) {
        assertEquals(expected.size(), actual.size());
        expected.sort(Comparator.comparingLong(x -> x.f0));
        actual.sort(Comparator.comparingLong(x -> x.f0));
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i).f0, actual.get(i).f0);
            assertArrayEquals(expected.get(i).f1, actual.get(i).f1, 1e-7);
        }
    }

    private static MockPojo[] sumPojo(MockPojo[] d1, MockPojo[] d2) {
        Preconditions.checkArgument(d1.length == d2.length);
        for (int i = 0; i < d1.length; i++) {
            d2[i].i += d1[i].i;
            d2[i].j += d1[i].j;
        }
        return d2;
    }

    private static Integer[] sumIntArray(Integer[] d1, Integer[] d2) {
        Preconditions.checkArgument(d1.length == d2.length);
        for (int i = 0; i < d1.length; i++) {
            d2[i] += d1[i];
        }
        return d2;
    }

    private static double[] trimToArray(SharedDoubleArray array) {
        return Arrays.copyOfRange(array.elements(), 0, array.size());
    }
}
