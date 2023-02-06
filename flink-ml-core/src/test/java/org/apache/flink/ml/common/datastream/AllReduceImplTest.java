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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.NumberSequenceIterator;

import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

/** Tests the {@link AllReduceImpl}. */
@RunWith(Enclosed.class)
public class AllReduceImplTest {

    private static final int parallelism = 4;

    private static final int chunkSize = AllReduceImpl.CHUNK_SIZE;

    private static final double TOLERANCE = 1e-7;

    /**
     * Parameterized test for {@link AllReduceImpl}. The test cases include:
     * <li>when there are no chunks.
     * <li>when the data is not enough for one chunk.
     * <li>when not every worker has one chunk to handle.
     * <li>when each worker needs to handle at least one chunk.
     */
    @RunWith(Parameterized.class)
    public static class ParameterizedTest {

        private static int numElements;
        private StreamExecutionEnvironment env;

        @Before
        public void before() {
            env = TestUtils.getExecutionEnvironment();
        }

        @Parameterized.Parameters
        public static Collection<Object[]> params() {
            return Arrays.asList(
                    new Object[][] {
                        {0},
                        {(int) (chunkSize * 0.5)},
                        {(int) (chunkSize * parallelism * 0.5)},
                        {(int) (chunkSize * parallelism * 1.5)}
                    });
        }

        public ParameterizedTest(int numElements) {
            ParameterizedTest.numElements = numElements;
        }

        @Test
        public void testAllReduce() throws Exception {
            DataStream<double[]> elements =
                    env.fromParallelCollection(
                                    new NumberSequenceIterator(1L, parallelism),
                                    BasicTypeInfo.LONG_TYPE_INFO)
                            .map(
                                    x -> {
                                        double[] res = new double[numElements];
                                        for (int i = 0; i < res.length; i++) {
                                            res[i] = i;
                                        }
                                        return res;
                                    });

            DataStreamUtils.allReduceSum(elements)
                    .addSink(
                            new SinkFunction<double[]>() {
                                @Override
                                public void invoke(double[] value, Context context) {
                                    assertEquals(numElements, value.length);
                                    for (int i = 0; i < value.length; i++) {
                                        assertEquals(i * parallelism, value[i], TOLERANCE);
                                    }
                                }
                            });

            env.execute();
        }
    }

    /** Non-parameterized test for {@link AllReduceImpl}. */
    public static class NonParameterizedTest {

        private StreamExecutionEnvironment env;

        @Before
        public void before() {
            env = TestUtils.getExecutionEnvironment();
        }

        @Test
        public void testAllReduceWithMoreThanOneArray() {
            try {
                DataStream<double[]> elements =
                        env.fromParallelCollection(
                                        new NumberSequenceIterator(1L, parallelism),
                                        BasicTypeInfo.LONG_TYPE_INFO)
                                .flatMap(
                                        new FlatMapFunction<Long, double[]>() {
                                            @Override
                                            public void flatMap(
                                                    Long value, Collector<double[]> out) {
                                                out.collect(new double[100]);
                                                out.collect(new double[100]);
                                            }
                                        });

                DataStreamUtils.allReduceSum(elements).addSink(new SinkFunction<double[]>() {});
                env.execute();
                fail();
            } catch (Exception e) {
                assertEquals(
                        "The input cannot contain more than one double array.",
                        e.getCause().getCause().getMessage());
            }
        }

        @Test
        public void testAllReduceWithDifferentLength() {
            try {
                DataStream<double[]> elements =
                        env.fromParallelCollection(
                                        new NumberSequenceIterator(1L, parallelism),
                                        BasicTypeInfo.LONG_TYPE_INFO)
                                .map(x -> new double[x.intValue()]);

                DataStreamUtils.allReduceSum(elements).addSink(new SinkFunction<double[]>() {});
                env.execute();
                fail();
            } catch (Exception e) {
                assertEquals(
                        "The input double array must have same length.",
                        e.getCause().getCause().getMessage());
            }
        }

        @Test
        public void testAllReduceWithEmptyInput() throws Exception {
            DataStream<double[]> elements =
                    env.fromParallelCollection(
                                    new NumberSequenceIterator(1L, parallelism),
                                    BasicTypeInfo.LONG_TYPE_INFO)
                            .flatMap((FlatMapFunction<Long, double[]>) (value, out) -> {})
                            .returns(PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO);
            Iterator<double[]> result = DataStreamUtils.allReduceSum(elements).executeAndCollect();
            assertFalse(result.hasNext());
        }
    }
}
