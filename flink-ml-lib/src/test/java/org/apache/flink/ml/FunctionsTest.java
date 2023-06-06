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

package org.apache.flink.ml;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.Before;
import org.junit.Test;

import javax.annotation.Nullable;

import java.util.Arrays;
import java.util.List;

import static org.apache.flink.ml.Functions.arrayToVector;
import static org.apache.flink.ml.Functions.vectorToArray;
import static org.apache.flink.table.api.Expressions.$;
import static org.junit.Assert.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/** Tests {@link Functions}. */
public class FunctionsTest extends AbstractTestBase {
    private static final List<double[]> doubleArrays =
            Arrays.asList(new double[] {0.0, 0.0}, new double[] {0.0, 1.0});

    private static final List<float[]> floatArrays =
            Arrays.asList(new float[] {0.0f, 0.0f}, new float[] {0.0f, 1.0f});

    private static final List<int[]> intArrays = Arrays.asList(new int[] {0, 0}, new int[] {0, 1});

    private static final List<long[]> longArrays =
            Arrays.asList(new long[] {0, 0}, new long[] {0, 1});

    private static final List<DenseIntDoubleVector> denseVectors =
            Arrays.asList(Vectors.dense(0.0, 0.0), Vectors.dense(0.0, 1.0));

    private static final List<SparseIntDoubleVector> sparseVectors =
            Arrays.asList(
                    Vectors.sparse(2, new int[0], new double[0]),
                    Vectors.sparse(2, new int[] {1}, new double[] {1.0}));

    private static final List<Vector> mixedVectors =
            Arrays.asList(
                    Vectors.dense(0.0, 0.0), Vectors.sparse(2, new int[] {1}, new double[] {1.0}));

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
    }

    @Test
    public void testVectorToArray() {
        testVectorToArray(denseVectors, null);
        testVectorToArray(sparseVectors, null);
        testVectorToArray(mixedVectors, VectorTypeInfo.INSTANCE);
    }

    private <T> void testVectorToArray(
            List<T> vectors, @Nullable TypeInformation<T> vectorTypeInformation) {
        Table inputTable;
        if (vectorTypeInformation == null) {
            inputTable = tEnv.fromDataStream(env.fromCollection(vectors));
        } else {
            inputTable = tEnv.fromDataStream(env.fromCollection(vectors, vectorTypeInformation));
        }
        inputTable = inputTable.as("vector");

        Table outputTable = inputTable.select(vectorToArray($("vector")).as("array"));

        List<Row> outputValues = IteratorUtils.toList(outputTable.execute().collect());

        assertEquals(outputValues.size(), doubleArrays.size());
        for (int i = 0; i < doubleArrays.size(); i++) {
            Double[] doubles = outputValues.get(i).getFieldAs("array");
            assertArrayEquals(doubleArrays.get(i), ArrayUtils.toPrimitive(doubles));
        }
    }

    @Test
    public void testArrayToVector() {
        testArrayToVector(doubleArrays);
        testArrayToVector(floatArrays);
        testArrayToVector(intArrays);
        testArrayToVector(longArrays);
    }

    private <T> void testArrayToVector(List<T> array) {
        Table inputTable = tEnv.fromDataStream(env.fromCollection(array)).as("array");

        Table outputTable = inputTable.select(arrayToVector($("array")).as("vector"));

        List<Row> outputValues = IteratorUtils.toList(outputTable.execute().collect());

        assertEquals(outputValues.size(), denseVectors.size());
        for (int i = 0; i < denseVectors.size(); i++) {
            DenseIntDoubleVector vector = outputValues.get(i).getFieldAs("vector");
            assertEquals(denseVectors.get(i), vector);
        }
    }
}
