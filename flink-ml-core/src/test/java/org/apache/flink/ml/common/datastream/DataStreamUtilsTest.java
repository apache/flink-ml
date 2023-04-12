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

package org.apache.flink.ml.common.datastream;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.CoGroupFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.NumberSequenceIterator;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/** Tests the {@link DataStreamUtils}. */
public class DataStreamUtilsTest {
    private StreamExecutionEnvironment env;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
    }

    @Test
    public void testCoGroupWithSingleParallelism() throws Exception {
        DataStream<Tuple2<Integer, Integer>> data1 =
                env.fromCollection(
                        Arrays.asList(Tuple2.of(1, 1), Tuple2.of(2, 2), Tuple2.of(3, 3)));
        DataStream<Tuple2<Integer, Double>> data2 =
                env.fromCollection(
                        Arrays.asList(
                                Tuple2.of(1, 1.5),
                                Tuple2.of(5, 5.5),
                                Tuple2.of(3, 3.5),
                                Tuple2.of(1, 2.5)));
        DataStream<Double> result =
                DataStreamUtils.coGroup(
                        data1,
                        data2,
                        (KeySelector<Tuple2<Integer, Integer>, Integer>) tuple -> tuple.f0,
                        (KeySelector<Tuple2<Integer, Double>, Integer>) tuple -> tuple.f0,
                        BasicTypeInfo.DOUBLE_TYPE_INFO,
                        new CoGroupFunction<
                                Tuple2<Integer, Integer>, Tuple2<Integer, Double>, Double>() {
                            @Override
                            public void coGroup(
                                    Iterable<Tuple2<Integer, Integer>> iterableA,
                                    Iterable<Tuple2<Integer, Double>> iterableB,
                                    Collector<Double> collector) {
                                List<Tuple2<Integer, Integer>> valuesA =
                                        IteratorUtils.toList(iterableA.iterator());
                                List<Tuple2<Integer, Double>> valuesB =
                                        IteratorUtils.toList(iterableB.iterator());

                                double sum = 0;
                                for (Tuple2<Integer, Integer> value : valuesA) {
                                    sum += value.f1;
                                }
                                for (Tuple2<Integer, Double> value : valuesB) {
                                    sum += value.f1;
                                }
                                collector.collect(sum);
                            }
                        });

        List<Double> resultValues = IteratorUtils.toList(result.executeAndCollect());
        double[] resultPrimitiveValues =
                resultValues.stream().mapToDouble(Double::doubleValue).toArray();
        double[] expectedResult = new double[] {5.0, 2.0, 6.5, 5.5};
        assertArrayEquals(expectedResult, resultPrimitiveValues, 1e-5);
    }

    @Test
    public void testCoGroupWithMultiParallelism() throws Exception {
        DataStream<Long> data1 =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 10L), Types.LONG);
        DataStream<Long> data2 =
                env.fromParallelCollection(new NumberSequenceIterator(6L, 16L), Types.LONG);

        DataStream<Long> result =
                DataStreamUtils.coGroup(
                        data1,
                        data2,
                        (KeySelector<Long, Long>) v -> v / 2,
                        (KeySelector<Long, Long>) v -> v / 2,
                        BasicTypeInfo.LONG_TYPE_INFO,
                        new CoGroupFunction<Long, Long, Long>() {
                            @Override
                            public void coGroup(
                                    Iterable<Long> iterableA,
                                    Iterable<Long> iterableB,
                                    Collector<Long> collector) {
                                List<Long> valuesA = IteratorUtils.toList(iterableA.iterator());
                                List<Long> valuesB = IteratorUtils.toList(iterableB.iterator());
                                long sum = 0;
                                for (Long value : valuesA) {
                                    sum += value;
                                }
                                for (Long value : valuesB) {
                                    sum += value;
                                }
                                collector.collect(sum);
                            }
                        });

        List<Long> resultValues = IteratorUtils.toList(result.executeAndCollect());
        long[] resultPrimitiveValues = resultValues.stream().mapToLong(Long::longValue).toArray();
        Arrays.sort(resultPrimitiveValues);
        long[] expectedResult = new long[] {1, 5, 9, 16, 25, 26, 29, 31, 34};
        assertArrayEquals(expectedResult, resultPrimitiveValues);
    }

    @Test
    public void testMapPartition() throws Exception {
        DataStream<Long> dataStream =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 19L), Types.LONG);
        DataStream<Integer> countsPerPartition =
                DataStreamUtils.mapPartition(dataStream, new TestMapPartitionFunc());
        List<Integer> counts = IteratorUtils.toList(countsPerPartition.executeAndCollect());
        assertArrayEquals(
                new int[] {5, 5, 5, 5}, counts.stream().mapToInt(Integer::intValue).toArray());
    }

    @Test
    public void testReduce() throws Exception {
        DataStream<Long> dataStream =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 19L), Types.LONG);
        DataStream<Long> result =
                DataStreamUtils.reduce(dataStream, (ReduceFunction<Long>) Long::sum);
        List<Long> sum = IteratorUtils.toList(result.executeAndCollect());
        assertArrayEquals(new long[] {190L}, sum.stream().mapToLong(Long::longValue).toArray());
    }

    @Test
    public void testAggregate() throws Exception {
        DataStream<Long> dataStream =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 19L), Types.LONG);
        DataStream<String> result = DataStreamUtils.aggregate(dataStream, new TestAggregateFunc());
        List<String> stringSum = IteratorUtils.toList(result.executeAndCollect());
        assertEquals(1, stringSum.size());
        assertEquals("190", stringSum.get(0));
    }

    @Test
    public void testAggregateWithNonNeutralInitialAccumulator() throws Exception {
        DataStream<Long> dataStream =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 19L), Types.LONG);
        DataStream<String> result =
                DataStreamUtils.aggregate(
                        dataStream, new TestAggregateFuncWithNonNeutralInitialAccumulator());
        List<String> stringSum = IteratorUtils.toList(result.executeAndCollect());
        assertEquals(1, stringSum.size());
        assertEquals(Integer.toString(190 + env.getParallelism()), stringSum.get(0));

        env.setParallelism(env.getParallelism() + 1);
        dataStream = env.fromParallelCollection(new NumberSequenceIterator(0L, 19L), Types.LONG);
        result =
                DataStreamUtils.aggregate(
                        dataStream, new TestAggregateFuncWithNonNeutralInitialAccumulator());
        stringSum = IteratorUtils.toList(result.executeAndCollect());
        assertEquals(1, stringSum.size());
        assertEquals(Integer.toString(190 + env.getParallelism()), stringSum.get(0));
    }

    @Test
    public void testSample() throws Exception {
        int numSamples = 10;
        int[] totalMinusOneChoices = new int[] {0, 5, 9, 10, 11, 20, 30, 40, 200};
        for (int totalMinusOne : totalMinusOneChoices) {
            DataStream<Long> dataStream =
                    env.fromParallelCollection(
                            new NumberSequenceIterator(0L, totalMinusOne), Types.LONG);
            DataStream<Long> result = DataStreamUtils.sample(dataStream, numSamples, 0);
            //noinspection unchecked
            List<String> sampled = IteratorUtils.toList(result.executeAndCollect());
            assertEquals(Math.min(numSamples, totalMinusOne + 1), sampled.size());
        }
    }

    @Test
    public void testGenerateBatchData() throws Exception {
        DataStream<Long> dataStream =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 19L), Types.LONG);
        DataStream<Long[]> result = DataStreamUtils.generateBatchData(dataStream, 2, 4);
        List<Long[]> batches = IteratorUtils.toList(result.executeAndCollect());
        for (Long[] batch : batches) {
            assertEquals(2, batch.length);
        }
        assertEquals(10, batches.size());
    }

    /** A simple implementation for a {@link MapPartitionFunction}. */
    private static class TestMapPartitionFunc extends RichMapPartitionFunction<Long, Integer> {

        public void mapPartition(Iterable<Long> values, Collector<Integer> out) {
            assertNotNull(getRuntimeContext());
            int cnt = 0;
            for (long ignored : values) {
                cnt++;
            }
            out.collect(cnt);
        }
    }

    /** A simple implementation for {@link AggregateFunction}. */
    private static class TestAggregateFunc implements AggregateFunction<Long, Long, String> {
        @Override
        public Long createAccumulator() {
            return 0L;
        }

        @Override
        public Long add(Long element, Long acc) {
            return element + acc;
        }

        @Override
        public String getResult(Long acc) {
            return String.valueOf(acc);
        }

        @Override
        public Long merge(Long acc1, Long acc2) {
            return acc1 + acc2;
        }
    }

    /**
     * An extension for {@link TestAggregateFunc} that provides a non-neutral initial accumulator.
     */
    private static class TestAggregateFuncWithNonNeutralInitialAccumulator
            extends TestAggregateFunc {
        @Override
        public Long createAccumulator() {
            return 1L;
        }
    }
}
