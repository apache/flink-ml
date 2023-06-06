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

package org.apache.flink.ml.feature;

import org.apache.flink.ml.feature.elementwiseproduct.ElementwiseProduct;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

/** Tests {@link ElementwiseProduct}. */
public class ElementwiseProductTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            0,
                            Vectors.dense(2.1, 3.1),
                            Vectors.sparse(5, new int[] {3}, new double[] {1.0})),
                    Row.of(
                            1,
                            Vectors.dense(1.1, 3.3),
                            Vectors.sparse(
                                    5, new int[] {4, 2, 3, 1}, new double[] {4.0, 2.0, 3.0, 1.0})),
                    Row.of(2, null, null));

    private static final double[] EXPECTED_OUTPUT_DENSE_VEC_ARRAY_1 = new double[] {2.31, 3.41};
    private static final double[] EXPECTED_OUTPUT_DENSE_VEC_ARRAY_2 = new double[] {1.21, 3.63};

    private static final int EXPECTED_OUTPUT_SPARSE_VEC_SIZE_1 = 5;
    private static final int[] EXPECTED_OUTPUT_SPARSE_VEC_INDICES_1 = new int[] {3};
    private static final double[] EXPECTED_OUTPUT_SPARSE_VEC_VALUES_1 = new double[] {0.0};

    private static final int EXPECTED_OUTPUT_SPARSE_VEC_SIZE_2 = 5;
    private static final int[] EXPECTED_OUTPUT_SPARSE_VEC_INDICES_2 = new int[] {1, 2, 3, 4};
    private static final double[] EXPECTED_OUTPUT_SPARSE_VEC_VALUES_2 =
            new double[] {1.1, 0.0, 0.0, 0.0};

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT_DATA);
        inputDataTable = tEnv.fromDataStream(dataStream).as("id", "vec", "sparseVec");
    }

    private void verifyOutputResult(Table output, String outputCol, boolean isSparse)
            throws Exception {
        DataStream<Row> dataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(3, results.size());
        for (Row result : results) {
            if (result.getField(0) == (Object) 0) {
                if (isSparse) {
                    SparseIntDoubleVector sparseVector =
                            (SparseIntDoubleVector) result.getField(outputCol);
                    assertEquals(EXPECTED_OUTPUT_SPARSE_VEC_SIZE_1, sparseVector.size().intValue());
                    assertArrayEquals(EXPECTED_OUTPUT_SPARSE_VEC_INDICES_1, sparseVector.indices);
                    assertArrayEquals(
                            EXPECTED_OUTPUT_SPARSE_VEC_VALUES_1, sparseVector.values, 1.0e-5);
                } else {
                    assertArrayEquals(
                            EXPECTED_OUTPUT_DENSE_VEC_ARRAY_1,
                            ((DenseIntDoubleVector) result.getField(outputCol)).values,
                            1.0e-5);
                }
            } else if (result.getField(0) == (Object) 1) {
                if (isSparse) {
                    SparseIntDoubleVector sparseVector =
                            (SparseIntDoubleVector) result.getField(outputCol);
                    assertEquals(EXPECTED_OUTPUT_SPARSE_VEC_SIZE_2, sparseVector.size().intValue());
                    assertArrayEquals(EXPECTED_OUTPUT_SPARSE_VEC_INDICES_2, sparseVector.indices);
                    assertArrayEquals(
                            EXPECTED_OUTPUT_SPARSE_VEC_VALUES_2, sparseVector.values, 1.0e-5);
                } else {
                    assertArrayEquals(
                            EXPECTED_OUTPUT_DENSE_VEC_ARRAY_2,
                            ((DenseIntDoubleVector) result.getField(outputCol)).values,
                            1.0e-5);
                }
            } else if (result.getField(0) == (Object) 2) {
                assertNull(result.getField(outputCol));
            } else {
                throw new UnsupportedOperationException("Input data id not exists.");
            }
        }
    }

    @Test
    public void testParam() {
        ElementwiseProduct elementwiseProduct = new ElementwiseProduct();
        assertEquals("output", elementwiseProduct.getOutputCol());
        assertEquals("input", elementwiseProduct.getInputCol());

        elementwiseProduct
                .setInputCol("vec")
                .setOutputCol("outputVec")
                .setScalingVec(Vectors.dense(1.0, 2.0, 3.0));
        assertEquals("vec", elementwiseProduct.getInputCol());
        assertEquals(Vectors.dense(1.0, 2.0, 3.0), elementwiseProduct.getScalingVec());
        assertEquals("outputVec", elementwiseProduct.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        ElementwiseProduct elementwiseProduct =
                new ElementwiseProduct()
                        .setInputCol("vec")
                        .setOutputCol("outputVec")
                        .setScalingVec(Vectors.dense(1.0, 2.0, 3.0));
        Table output = elementwiseProduct.transform(inputDataTable)[0];
        assertEquals(
                Arrays.asList("id", "vec", "sparseVec", "outputVec"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testSaveLoadAndTransformDense() throws Exception {
        ElementwiseProduct elementwiseProduct =
                new ElementwiseProduct()
                        .setInputCol("vec")
                        .setOutputCol("outputVec")
                        .setScalingVec(Vectors.dense(1.1, 1.1));
        ElementwiseProduct loadedElementwiseProduct =
                TestUtils.saveAndReload(
                        tEnv,
                        elementwiseProduct,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        ElementwiseProduct::load);
        Table output = loadedElementwiseProduct.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedElementwiseProduct.getOutputCol(), false);
    }

    @Test
    public void testVectorSizeNotEquals() {
        try {
            ElementwiseProduct elementwiseProduct =
                    new ElementwiseProduct()
                            .setInputCol("vec")
                            .setOutputCol("outputVec")
                            .setScalingVec(Vectors.dense(1.1, 1.1, 2.0));
            Table output = elementwiseProduct.transform(inputDataTable)[0];
            DataStream<Row> dataStream = tEnv.toDataStream(output);
            IteratorUtils.toList(dataStream.executeAndCollect());
            fail();
        } catch (Exception e) {
            assertEquals(
                    "The scaling vector size is 3, which is not equal input vector size(2).",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testSaveLoadAndTransformSparse() throws Exception {
        ElementwiseProduct elementwiseProduct =
                new ElementwiseProduct()
                        .setInputCol("sparseVec")
                        .setOutputCol("outputVec")
                        .setScalingVec(
                                Vectors.sparse(5, new int[] {0, 1}, new double[] {1.1, 1.1}));
        ElementwiseProduct loadedElementwiseProduct =
                TestUtils.saveAndReload(
                        tEnv,
                        elementwiseProduct,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        ElementwiseProduct::load);
        Table output = loadedElementwiseProduct.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedElementwiseProduct.getOutputCol(), true);
    }
}
