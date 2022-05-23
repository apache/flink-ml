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

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.vectorassembler.VectorAssembler;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
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
import static org.junit.Assert.assertNull;

/** Tests VectorAssembler. */
public class VectorAssemblerTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

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
                    Row.of(2, null, null, null));

    private static final SparseVector EXPECTED_OUTPUT_DATA_1 =
            Vectors.sparse(8, new int[] {0, 1, 2, 6}, new double[] {2.1, 3.1, 1.0, 1.0});
    private static final DenseVector EXPECTED_OUTPUT_DATA_2 =
            Vectors.dense(2.1, 3.1, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0);

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT_DATA);
        inputDataTable = tEnv.fromDataStream(dataStream).as("id", "vec", "num", "sparseVec");
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
            } else {
                assertNull(result.getField(outputCol));
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
                .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        assertArrayEquals(new String[] {"vec", "num", "sparseVec"}, vectorAssembler.getInputCols());
        assertEquals(HasHandleInvalid.SKIP_INVALID, vectorAssembler.getHandleInvalid());
        assertEquals("assembledVec", vectorAssembler.getOutputCol());
    }

    @Test
    public void testKeepInvalid() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        Table output = vectorAssembler.transform(inputDataTable)[0];
        assertEquals(
                Arrays.asList("id", "vec", "num", "sparseVec", "assembledVec"),
                output.getResolvedSchema().getColumnNames());
        verifyOutputResult(output, vectorAssembler.getOutputCol(), 3);
    }

    @Test
    public void testErrorInvalid() {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setHandleInvalid(HasHandleInvalid.ERROR_INVALID);
        try {
            Table outputTable = vectorAssembler.transform(inputDataTable)[0];
            outputTable.execute().collect().next();
            Assert.fail("Expected IllegalArgumentException");
        } catch (Throwable e) {
            assertEquals(
                    "Input column value should not be null.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testSkipInvalid() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        Table output = vectorAssembler.transform(inputDataTable)[0];
        assertEquals(
                Arrays.asList("id", "vec", "num", "sparseVec", "assembledVec"),
                output.getResolvedSchema().getColumnNames());
        verifyOutputResult(output, vectorAssembler.getOutputCol(), 2);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        VectorAssembler loadedVectorAssembler =
                TestUtils.saveAndReload(
                        tEnv, vectorAssembler, TEMPORARY_FOLDER.newFolder().getAbsolutePath());
        Table output = loadedVectorAssembler.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedVectorAssembler.getOutputCol(), 2);
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        inputDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, inputDataTable);
        assertArrayEquals(
                new Class<?>[] {
                    Integer.class, SparseVector.class, Integer.class, SparseVector.class
                },
                TestUtils.getColumnDataTypes(inputDataTable));

        VectorAssembler vectorAssembler =
                new VectorAssembler()
                        .setInputCols("vec", "num", "sparseVec")
                        .setOutputCol("assembledVec")
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        VectorAssembler loadedVectorAssembler =
                TestUtils.saveAndReload(
                        tEnv, vectorAssembler, TEMPORARY_FOLDER.newFolder().getAbsolutePath());
        Table output = loadedVectorAssembler.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedVectorAssembler.getOutputCol(), 2);
    }
}
