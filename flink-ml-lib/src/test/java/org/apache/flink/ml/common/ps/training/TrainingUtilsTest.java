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

package org.apache.flink.ml.common.ps.training;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.util.ResettableIterator;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableSupplier;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/** Tests {@link TrainingUtils}. */
public class TrainingUtilsTest {
    private static final int NUM_WORKERS = 2;
    private static final int NUM_SERVERS = 2;
    private static final int MAX_ITER = 3;
    private static final int NUM_COLUMNS_PER_KEY = 2;

    private DataStream<Long> maxKey;
    private DataStream<DenseIntDoubleVector> inputData;

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        env.setParallelism(NUM_WORKERS);
        maxKey = env.fromElements(3L);
        inputData =
                env.fromCollection(
                                Arrays.asList(
                                        Vectors.dense(1, 1, 1, 1),
                                        Vectors.dense(2, 2, 2, 2),
                                        Vectors.dense(3, 3, 3, 3),
                                        Vectors.dense(4, 4, 4, 4)))
                        .map(x -> x, DenseIntDoubleVectorTypeInfo.INSTANCE);
    }

    @Test
    public void test() throws Exception {
        ExecutionConfig config = maxKey.getExecutionEnvironment().getConfig();

        TypeSerializer<MockPojo> pojoDemoTypeSerializer =
                Types.POJO(MockPojo.class).createSerializer(config);

        MockSession mockSession =
                new MockSession(
                        DenseIntDoubleVectorTypeInfo.INSTANCE,
                        Collections.singletonList(
                                new OutputTag<>("AllReduceOutputTag", Types.POJO(MockPojo.class))));

        IterationStageList<MockSession> stageList =
                new IterationStageList<>(mockSession)
                        .addStage(new MockComputePullIndicesStage())
                        .addStage(
                                new PullStage(
                                        (SerializableSupplier<long[]>)
                                                () -> mockSession.pullIndices,
                                        (SerializableConsumer<double[]>)
                                                x -> mockSession.pulledValues = x))
                        .addStage(
                                new AllReduceStage<>(
                                        (SerializableSupplier<MockPojo[]>)
                                                () -> mockSession.toAllReduce,
                                        (SerializableConsumer<MockPojo[]>)
                                                x -> mockSession.toAllReduce = x,
                                        (ReduceFunction<MockPojo[]>) TrainingUtilsTest::sumPojo,
                                        pojoDemoTypeSerializer))
                        .addStage(new MockComputePushValuesStage(NUM_COLUMNS_PER_KEY))
                        .addStage(
                                new PushStage(
                                        (SerializableSupplier<long[]>)
                                                () -> mockSession.pushIndices,
                                        (SerializableSupplier<double[]>)
                                                () -> mockSession.pushValues))
                        .setTerminationCriteria(context -> context.iterationId >= MAX_ITER);

        DataStreamList resultList =
                TrainingUtils.train(
                        inputData,
                        stageList,
                        maxKey,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        new MockModelUpdater(NUM_COLUMNS_PER_KEY),
                        NUM_SERVERS);

        // Verifies the model data.
        DataStream<Tuple3<Long, Long, double[]>> modelSegments = resultList.get(0);
        List<Tuple3<Long, Long, double[]>> collectedModelPieces =
                IteratorUtils.toList(modelSegments.executeAndCollect());
        Assert.assertEquals(NUM_SERVERS, collectedModelPieces.size());
        collectedModelPieces.sort(Comparator.comparing(o -> o.f0));

        double[] result = new double[4 * NUM_COLUMNS_PER_KEY];
        double[] expectedResult = new double[4 * NUM_COLUMNS_PER_KEY];
        Arrays.fill(expectedResult, 0, NUM_COLUMNS_PER_KEY, 35.0);
        Arrays.fill(expectedResult, 3 * NUM_COLUMNS_PER_KEY, 4 * NUM_COLUMNS_PER_KEY, 35.0);
        for (Tuple3<Long, Long, double[]> modelPiece : collectedModelPieces) {
            int startIndex = (int) (long) modelPiece.f0 * NUM_COLUMNS_PER_KEY;
            double[] pieceCoeff = modelPiece.f2;
            System.arraycopy(pieceCoeff, 0, result, startIndex, pieceCoeff.length);
        }
        Assert.assertArrayEquals(expectedResult, result, 1e-7);

        // Verifies the all reduce result from worker output.
        DataStream<MockPojo> allReduceResult = resultList.get(1);
        allReduceResult.getTransformation().setParallelism(1);
        List<MockPojo> collectedPojo = IteratorUtils.toList(allReduceResult.executeAndCollect());
        List<MockPojo> expectedPojo =
                Arrays.asList(new MockPojo(1, 0), new MockPojo(2, 0), new MockPojo(4, 0));
        TestBaseUtils.compareResultCollections(
                expectedPojo,
                collectedPojo,
                new Comparator<MockPojo>() {
                    @Override
                    public int compare(MockPojo o1, MockPojo o2) {
                        return Integer.compare(o1.i, o2.i);
                    }
                });
    }

    private static MockPojo[] sumPojo(MockPojo[] d1, MockPojo[] d2) {
        Preconditions.checkArgument(d1.length == d2.length);
        for (int i = 0; i < d1.length; i++) {
            d2[i].i += d1[i].i;
            d2[i].j += d1[i].j;
        }
        return d2;
    }

    private static class MockSession extends MiniBatchMLSession<DenseIntDoubleVector> {

        public MockPojo[] toAllReduce;
        private ProxySideOutput output;

        @Override
        public void setOutput(ProxySideOutput collector) {
            this.output = collector;
        }

        public MockSession(
                TypeInformation<DenseIntDoubleVector> typeInformation,
                List<OutputTag<?>> outputTags) {
            super(0, typeInformation, outputTags);
        }
    }

    /** Pulls the 0-th and 3-th dimension of the model from servers. */
    private static class MockComputePullIndicesStage extends ProcessStage<MockSession> {

        @Override
        public void process(MockSession context) {
            context.pullIndices = new long[] {0, 3};
            if (context.toAllReduce == null) {
                context.toAllReduce = new MockPojo[1];
                context.toAllReduce[0] = new MockPojo(1, 0);
            }
            if (context.workerId == 0) {
                context.output.output(
                        (OutputTag<MockPojo>) context.getOutputTags().get(0),
                        new StreamRecord(context.toAllReduce[0]));
            }
        }
    }

    /**
     * Adds the 0-th and 3-th dimension of all training data to the model and pushes it to servers.
     */
    private static class MockComputePushValuesStage extends ProcessStage<MockSession> {
        private final int numCols;

        public MockComputePushValuesStage(int numCols) {
            this.numCols = numCols;
        }

        @Override
        public void process(MockSession context) throws Exception {
            long[] indices = context.pullIndices;
            double[] values = context.pulledValues;
            ResettableIterator<DenseIntDoubleVector> data = context.inputData;
            while (data.hasNext()) {
                double[] d = data.next().values;
                for (int i = 0; i < indices.length; i++) {
                    double v = d[(int) indices[i]];
                    for (int j = 0; j < numCols; j++) {
                        values[i * numCols + j] += v;
                    }
                }
            }
            data.reset();

            BLAS.scal(1.0 / context.numWorkers, new DenseIntDoubleVector(values));

            context.pushIndices = indices;
            context.pushValues = values;
        }
    }

    /** The logic on servers. */
    private static class MockModelUpdater implements ModelUpdater<Tuple3<Long, Long, double[]>> {
        private final int numDoublesPerKey;
        private long startIndex;
        private long endIndex;
        private double[] model;

        private ListState<Long> boundaryState;
        private ListState<double[]> modelDataState;

        public MockModelUpdater(int numDoublesPerKey) {
            this.numDoublesPerKey = numDoublesPerKey;
        }

        @Override
        public void open(long startKeyIndex, long endKeyIndex) {
            this.startIndex = startKeyIndex;
            this.endIndex = endKeyIndex;
            this.model = new double[(int) (endKeyIndex - startKeyIndex) * numDoublesPerKey];
        }

        @Override
        public void update(long[] keys, double[] values) {
            Preconditions.checkState(keys.length * numDoublesPerKey == values.length);
            for (int i = 0; i < keys.length; i++) {
                int index = (int) (keys[i] - startIndex);
                for (int j = 0; j < numDoublesPerKey; j++) {
                    model[index * numDoublesPerKey + j] += values[i * numDoublesPerKey + j];
                }
            }
        }

        @Override
        public double[] get(long[] keys) {
            double[] values = new double[keys.length * numDoublesPerKey];
            for (int i = 0; i < keys.length; i++) {
                int index = (int) (keys[i] - startIndex);
                for (int j = 0; j < numDoublesPerKey; j++) {
                    values[i * numDoublesPerKey + j] += model[index * numDoublesPerKey + j];
                }
            }
            return values;
        }

        @Override
        public Iterator<Tuple3<Long, Long, double[]>> getModelSegments() {
            return Collections.singleton(Tuple3.of(startIndex, endIndex, model)).iterator();
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            boundaryState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("BoundaryState", Types.LONG));

            Iterator<Long> iterator = boundaryState.get().iterator();
            if (iterator.hasNext()) {
                startIndex = iterator.next();
                endIndex = iterator.next();
            }

            modelDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "modelDataState",
                                            PrimitiveArrayTypeInfo
                                                    .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
            Iterator<double[]> modelData = modelDataState.get().iterator();
            if (modelData.hasNext()) {
                model = modelData.next();
            }
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            if (model != null) {
                boundaryState.clear();
                boundaryState.add(startIndex);
                boundaryState.add(endIndex);

                modelDataState.clear();
                modelDataState.add(model);
            }
        }
    }

    /** Mock pojo class to test all reduce. */
    public static class MockPojo {
        public int i;
        public int j;

        public MockPojo(int i, int j) {
            this.i = i;
            this.j = j;
        }

        public MockPojo() {}

        @Override
        public String toString() {
            return i + "-" + j;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof MockPojo) {
                MockPojo other = (MockPojo) obj;
                return i == other.i && j == other.j;
            }
            return false;
        }
    }
}
