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

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.vectorassembler.VectorAssembler;
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
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests VectorAssembler. */
public class VectorAssemblerTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;
    private Table inputNullDataTable;
    private Table inputNanDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            0,
                            Vectors.dense(2.1, 3.1),
                            1.0,
                            Vectors.sparse(5, new int[] {3}, new double[] {1.0})),
                    Row.of(
                            1,
                            Vectors.dense(2.1, 3.1),
                            1.0,
                            Vectors.sparse(
                                    5, new int[] {4, 2, 3, 1}, new double[] {4.0, 2.0, 3.0, 1.0})),
                    Row.of(
                            2,
                            Vectors.dense(2.0, 2.1),
                            1.0,
                            Vectors.sparse(
                                    5, new int[] {4, 2, 3, 1}, new double[] {4.0, 2.0, 3.0, 1.0})));

    private static final List<Row> INPUT_NAN_DATA =
            Arrays.asList(
                    Row.of(
                            0,
                            Vectors.dense(2.1, 3.1),
                            1.0,
                            Vectors.sparse(5, new int[] {3}, new double[] {1.0})),
                    Row.of(
                            1,
                            Vectors.dense(2.1, 3.1),
                            1.0,
                            Vectors.sparse(
                                    5, new int[] {4, 2, 3, 1}, new double[] {4.0, 2.0, 3.0, 1.0})),
                    Row.of(
                            2,
                            Vectors.dense(2.0, 2.1),
                            Double.NaN,
                            Vectors.sparse(
                                    5, new int[] {4, 2, 3, 1}, new double[] {4.0, 2.0, 3.0, 1.0})));

    private static final List<Row> INPUT_NULL_DATA =
            Arrays.asList(
                    Row.of(
                            0,
                            Vectors.dense(2.1, 3.1),
                            1.0,
                            Vectors.sparse(5, new int[] {3}, new double[] {1.0})),
                    Row.of(
                            1,
                            null,
                            1.0,
                            Vectors.sparse(
                                    5, new int[] {4, 2, 3, 1}, new double[] {4.0, 2.0, 3.0, 1.0})),
                    Row.of(
                            2,
                            null,
                            1.0,
                            Vectors.sparse(
                                    5, new int[] {4, 2, 3, 1}, new double[] {4.0, 2.0, 3.0, 1.0})));

    private static final SparseIntDoubleVector EXPECTED_OUTPUT_DATA_1 =
            Vectors.sparse(8, new int[] {0, 1, 2, 6}, new double[] {2.1, 3.1, 1.0, 1.0});
    private static final DenseIntDoubleVector EXPECTED_OUTPUT_DATA_2 =
            Vectors.dense(2.1, 3.1, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0);
    private static final DenseIntDoubleVector EXPECTED_OUTPUT_DATA_3 =
            Vectors.dense(2.0, 2.1, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0);
    private static final DenseIntDoubleVector EXPECTED_OUTPUT_DATA_4 =
            Vectors.dense(Double.NaN, Double.NaN, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0);
    private static final DenseIntDoubleVector EXPECTED_OUTPUT_DATA_5 =
            Vectors.dense(2.0, 2.1, Double.NaN, 0.0, 1.0, 2.0, 3.0, 4.0);

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT_DATA);
        inputDataTable = tEnv.fromDataStream(dataStream).as("id", "vec", "num", "sparseVec");
        DataStream<Row> nullDataStream = env.fromCollection(INPUT_NULL_DATA);
        inputNullDataTable =
                tEnv.fromDataStream(nullDataStream).as("id", "vec", "num", "sparseVec");
        DataStream<Row> nanDataStream = env.fromCollection(INPUT_NAN_DATA);
        inputNanDataTable = tEnv.fromDataStream(nanDataStream).as("id", "vec", "num", "sparseVec");
    }

    private void verifyOutputResult(Table output, String outputCol, int outputSize)
            throws Exception {
        DataStream<Row> dataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(outputSize, results.size());

        for (Row result : results) {
            if (result.getField(0) == (Object) 0) {
                assertEquals(EXPECTED_OUTPUT_DATA_1, result.getField(outputCol));
            } else if (result.getField(0) == (Object) 1) {
                assertEquals(EXPECTED_OUTPUT_DATA_2, result.getField(outputCol));
            } else if (result.getField(0) == (Object) 2) {
                assertEquals(EXPECTED_OUTPUT_DATA_3, result.getField(outputCol));
            }
        }
    }

    @Test
    public void testParam() {
        VectorAssembler vectorAssembler = new VectorAssembler();
        assertEquals(HasHandleInvalid.ERROR_INVALID, vectorAssembler.getHandleInvalid());
        assertEquals("output", vectorAssembler.getOutputCol());

        vectorAssembler
                .setInputCols("vec", "num", "sparseVec")
                .setOutputCol("assembledVec")
                .setInputSizes(2, 1, 5)
                .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);

        assertArrayEquals(new String[] {"vec", "num", "sparseVec"}, vectorAssembler.getInputCols());
        assertEquals(HasHandleInvalid.SKIP_INVALID, vectorAssembler.getHandleInvalid());
        assertEquals("assembledVec", vectorAssembler.getOutputCol());
        assertArrayEquals(new Integer[] {2, 1, 5}, vectorAssembler.getInputSizes());
    }

    @Test
    public void testOutputSchema() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("num")
                        .setOutputCol("assembledVec")
                        .setInputSizes(1)
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        Table output = vectorAssembler.transform(inputDataTable)[0];

        assertEquals(
                Arrays.asList("id", "vec", "num", "sparseVec", "assembledVec"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testKeepInvalidWithNull() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        Table output = vectorAssembler.transform(inputNullDataTable)[0];

        DataStream<Row> dataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(3, results.size());

        String outputCol = vectorAssembler.getOutputCol();
        for (Row result : results) {
            if (result.getField(0) == (Object) 0) {
                assertEquals(EXPECTED_OUTPUT_DATA_1, result.getField(outputCol));
            } else if (result.getField(0) == (Object) 1) {
                assertEquals(EXPECTED_OUTPUT_DATA_4, result.getField(outputCol));
            } else if (result.getField(0) == (Object) 2) {
                assertEquals(EXPECTED_OUTPUT_DATA_4, result.getField(outputCol));
            }
        }
    }

    @Test
    public void testKeepInvalidWithNaN() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        Table output = vectorAssembler.transform(inputNanDataTable)[0];
        DataStream<Row> dataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(3, results.size());

        String outputCol = vectorAssembler.getOutputCol();
        for (Row result : results) {
            if (result.getField(0) == (Object) 0) {
                assertEquals(EXPECTED_OUTPUT_DATA_1, result.getField(outputCol));
            } else if (result.getField(0) == (Object) 1) {
                assertEquals(EXPECTED_OUTPUT_DATA_2, result.getField(outputCol));
            } else if (result.getField(0) == (Object) 2) {
                assertEquals(EXPECTED_OUTPUT_DATA_5, result.getField(outputCol));
            }
        }
    }

    @Test
    public void testKeepInvalidWithErrorSizes() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 4)
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        Table output = vectorAssembler.transform(inputDataTable)[0];
        verifyOutputResult(output, vectorAssembler.getOutputCol(), 3);
    }

    @Test
    public void testErrorInvalidWithNull() {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.ERROR_INVALID);

        try {
            Table outputTable = vectorAssembler.transform(inputNullDataTable)[0];
            outputTable.execute().collect().next();
            Assert.fail("Expected IllegalArgumentException");
        } catch (Throwable e) {
            assertEquals(
                    "Vector assembler failed with exception : java.lang.RuntimeException: "
                            + "Input column value is null. Please check the input data or using handleInvalid = 'keep'.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testErrorInvalidWithNaN() {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.ERROR_INVALID);

        try {
            Table outputTable = vectorAssembler.transform(inputNanDataTable)[0];
            outputTable.execute().collect().next();
            Assert.fail("Expected IllegalArgumentException");
        } catch (Throwable e) {
            assertEquals(
                    "Vector assembler failed with exception : java.lang.RuntimeException: Encountered NaN "
                            + "while assembling a row with handleInvalid = 'error'. Consider removing NaNs from "
                            + "dataset or using handleInvalid = 'keep' or 'skip'.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testErrorInvalidWithErrorSizes() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 4)
                        .setHandleInvalid(HasHandleInvalid.ERROR_INVALID);
        try {
            Table outputTable = vectorAssembler.transform(inputDataTable)[0];
            outputTable.execute().collect().next();
            Assert.fail("Expected IllegalArgumentException");
        } catch (Throwable e) {
            assertEquals(
                    "Vector assembler failed with exception : java.lang.IllegalArgumentException: "
                            + "Input vector/number size does not meet with expected. Expected size: 4, actual size: 5.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testSkipInvalidWithNull() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        Table output = vectorAssembler.transform(inputNullDataTable)[0];
        verifyOutputResult(output, vectorAssembler.getOutputCol(), 1);
    }

    @Test
    public void testSkipInvalidWithNaN() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        Table output = vectorAssembler.transform(inputNanDataTable)[0];

        verifyOutputResult(output, vectorAssembler.getOutputCol(), 2);
    }

    @Test
    public void testSkipInvalidWithErrorSizes() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 4)
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        Table output = vectorAssembler.transform(inputDataTable)[0];
        verifyOutputResult(output, vectorAssembler.getOutputCol(), 0);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);

        VectorAssembler loadedVectorAssembler =
                TestUtils.saveAndReload(
                        tEnv,
                        vectorAssembler,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        VectorAssembler::load);

        Table output = loadedVectorAssembler.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedVectorAssembler.getOutputCol(), 3);
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        inputDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, inputDataTable);
        assertArrayEquals(
                new Class<?>[] {
                    Integer.class,
                    SparseIntDoubleVector.class,
                    Integer.class,
                    SparseIntDoubleVector.class
                },
                TestUtils.getColumnDataTypes(inputDataTable));

        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setInputSizes(2, 1, 5)
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);

        VectorAssembler loadedVectorAssembler =
                TestUtils.saveAndReload(
                        tEnv,
                        vectorAssembler,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        VectorAssembler::load);

        Table output = loadedVectorAssembler.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedVectorAssembler.getOutputCol(), 3);
    }

    @Test
    public void testNumber2Vector() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("num")
                        .setOutputCol("assembledVec")
                        .setInputSizes(1)
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        Table output = vectorAssembler.transform(inputDataTable)[0];

        DataStream<Row> dataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        for (Row result : results) {
            if (result.getField(2) != null) {
                assertEquals(
                        result.getField(2), ((DenseIntDoubleVector) result.getField(4)).values[0]);
            }
        }
    }
}
