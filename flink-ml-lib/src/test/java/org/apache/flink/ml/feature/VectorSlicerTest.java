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

import org.apache.flink.ml.feature.vectorslicer.VectorSlicer;
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
import static org.junit.Assert.fail;

/** Tests {@link VectorSlicer}. */
public class VectorSlicerTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            0,
                            Vectors.dense(2.1, 3.1, 2.3, 3.4, 5.3, 5.1),
                            Vectors.sparse(5, new int[] {1, 3, 4}, new double[] {0.1, 0.2, 0.3})),
                    Row.of(
                            1,
                            Vectors.dense(2.3, 4.1, 1.3, 2.4, 5.1, 4.1),
                            Vectors.sparse(5, new int[] {1, 2, 4}, new double[] {0.1, 0.2, 0.3})));

    private static final DenseIntDoubleVector EXPECTED_OUTPUT_DATA_1 = Vectors.dense(2.1, 3.1, 2.3);
    private static final DenseIntDoubleVector EXPECTED_OUTPUT_DATA_2 = Vectors.dense(2.3, 4.1, 1.3);

    private static final SparseIntDoubleVector EXPECTED_OUTPUT_DATA_3 =
            Vectors.sparse(3, new int[] {1}, new double[] {0.1});
    private static final SparseIntDoubleVector EXPECTED_OUTPUT_DATA_4 =
            Vectors.sparse(3, new int[] {1, 2}, new double[] {0.1, 0.2});

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
        assertEquals(2, results.size());
        for (Row result : results) {
            if (result.getField(0) == (Object) 0) {
                if (isSparse) {
                    assertEquals(EXPECTED_OUTPUT_DATA_3, result.getField(outputCol));
                } else {
                    assertEquals(EXPECTED_OUTPUT_DATA_1, result.getField(outputCol));
                }
            } else if (result.getField(0) == (Object) 1) {
                if (isSparse) {
                    assertEquals(EXPECTED_OUTPUT_DATA_4, result.getField(outputCol));
                } else {
                    assertEquals(EXPECTED_OUTPUT_DATA_2, result.getField(outputCol));
                }
            } else {
                throw new RuntimeException("Result id value is error, it must be 0 or 1.");
            }
        }
    }

    @Test
    public void testParam() {
        VectorSlicer vectorSlicer = new VectorSlicer();
        assertEquals("input", vectorSlicer.getInputCol());
        assertEquals("output", vectorSlicer.getOutputCol());
        vectorSlicer.setInputCol("vec").setOutputCol("sliceVec").setIndices(0, 1, 2);
        assertEquals("vec", vectorSlicer.getInputCol());
        assertEquals("sliceVec", vectorSlicer.getOutputCol());
        assertArrayEquals(new Integer[] {0, 1, 2}, vectorSlicer.getIndices());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        VectorSlicer vectorSlicer =
                new VectorSlicer().setInputCol("vec").setOutputCol("sliceVec").setIndices(0, 1, 2);
        VectorSlicer loadedVectorSlicer =
                TestUtils.saveAndReload(
                        tEnv,
                        vectorSlicer,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        VectorSlicer::load);
        Table output = loadedVectorSlicer.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedVectorSlicer.getOutputCol(), false);
    }

    @Test
    public void testEmptyIndices() {
        try {
            VectorSlicer vectorSlicer =
                    new VectorSlicer().setInputCol("vec").setOutputCol("sliceVec").setIndices();
            vectorSlicer.transform(inputDataTable);
            fail();
        } catch (Exception e) {
            assertEquals("Parameter indices is given an invalid value {}", e.getMessage());
        }
    }

    @Test
    public void testIndicesLargerThanVectorSize() {
        try {
            VectorSlicer vectorSlicer =
                    new VectorSlicer()
                            .setInputCol("vec")
                            .setOutputCol("sliceVec")
                            .setIndices(1, 2, 10);
            Table output = vectorSlicer.transform(inputDataTable)[0];
            DataStream<Row> dataStream = tEnv.toDataStream(output);
            IteratorUtils.toList(dataStream.executeAndCollect());
            fail();
        } catch (Exception e) {
            assertEquals(
                    "Index value 10 is greater than vector size:6",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testIndicesSmallerThanZero() {
        try {
            new VectorSlicer().setInputCol("vec").setOutputCol("sliceVec").setIndices(1, -2);
            fail();
        } catch (Exception e) {
            assertEquals("Parameter indices is given an invalid value {1,-2}", e.getMessage());
        }
    }

    @Test
    public void testDuplicateIndices() {
        try {
            new VectorSlicer().setInputCol("vec").setOutputCol("sliceVec").setIndices(1, 1, 3);
            fail();
        } catch (Exception e) {
            assertEquals("Parameter indices is given an invalid value {1,1,3}", e.getMessage());
        }
    }

    @Test
    public void testDenseTransform() throws Exception {
        VectorSlicer vectorSlicer =
                new VectorSlicer().setInputCol("vec").setOutputCol("sliceVec").setIndices(0, 1, 2);

        Table output = vectorSlicer.transform(inputDataTable)[0];
        verifyOutputResult(output, vectorSlicer.getOutputCol(), false);
    }

    @Test
    public void testDenseTransformWithUnorderedIndices() throws Exception {
        VectorSlicer vectorSlicer =
                new VectorSlicer().setInputCol("vec").setOutputCol("sliceVec").setIndices(0, 2, 1);

        Table output = vectorSlicer.transform(inputDataTable)[0];
        DataStream<Row> dataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(2, results.size());
        for (Row result : results) {
            if (result.getField(0) == (Object) 0) {
                assertEquals(
                        Vectors.dense(2.1, 2.3, 3.1), result.getField(vectorSlicer.getOutputCol()));

            } else if (result.getField(0) == (Object) 1) {
                assertEquals(
                        Vectors.dense(2.3, 1.3, 4.1), result.getField(vectorSlicer.getOutputCol()));
            } else {
                throw new RuntimeException("Result id value is error, it must be 0 or 1.");
            }
        }
    }

    @Test
    public void testSparseTransform() throws Exception {
        VectorSlicer vectorSlicer =
                new VectorSlicer()
                        .setInputCol("sparseVec")
                        .setOutputCol("sliceVec")
                        .setIndices(0, 1, 2);
        Table output = vectorSlicer.transform(inputDataTable)[0];
        verifyOutputResult(output, vectorSlicer.getOutputCol(), true);
    }
}
