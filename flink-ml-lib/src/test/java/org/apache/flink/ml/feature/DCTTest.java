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

import org.apache.flink.ml.feature.dct.DCT;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Tests the {@link DCT}. */
public class DCTTest extends AbstractTestBase {
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    private static final List<IntDoubleVector> inputData =
            Arrays.asList(Vectors.dense(1.0, 1.0, 1.0, 1.0), Vectors.dense(1.0, 0.0, -1.0, 0.0));

    private static final List<Row> expectedForwardOutputData =
            Arrays.asList(
                    Row.of(Vectors.dense(1.0, 1.0, 1.0, 1.0), Vectors.dense(2.0, 0.0, 0.0, 0.0)),
                    Row.of(
                            Vectors.dense(1.0, 0.0, -1.0, 0.0),
                            Vectors.dense(0.0, 0.924, 1.0, -0.383)));

    private static final List<Row> expectedInverseOutputData =
            Arrays.asList(
                    Row.of(
                            Vectors.dense(1.0, 1.0, 1.0, 1.0),
                            Vectors.dense(1.924, -0.383, 0.383, 0.076)),
                    Row.of(Vectors.dense(1.0, 0.0, -1.0, 0.0), Vectors.dense(0.0, 1.0, 1.0, 0.0)));

    private Table inputTable;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        inputTable = tEnv.fromDataStream(env.fromCollection(inputData)).as("input");
    }

    @Test
    public void testParam() {
        DCT dct = new DCT();

        assertEquals("input", dct.getInputCol());
        assertEquals("output", dct.getOutputCol());
        assertFalse(dct.getInverse());

        dct.setInputCol("test_input").setOutputCol("test_output").setInverse(true);

        assertEquals("test_input", dct.getInputCol());
        assertEquals("test_output", dct.getOutputCol());
        assertTrue(dct.getInverse());
    }

    @Test
    public void testOutputSchema() {
        Table inputTable =
                tEnv.fromDataStream(env.fromElements(Row.of(Vectors.dense(0.0), "")))
                        .as("test_input", "dummy_input");

        DCT dct = new DCT().setInputCol("test_input").setOutputCol("test_output");

        Table outputTable = dct.transform(inputTable)[0];

        assertEquals(
                Arrays.asList(dct.getInputCol(), "dummy_input", dct.getOutputCol()),
                outputTable.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransformForward() {
        DCT dct = new DCT();
        Table outputTable = dct.transform(inputTable)[0];

        verifyTransformResult(
                outputTable, expectedForwardOutputData, dct.getInputCol(), dct.getOutputCol());
    }

    @Test
    public void testTransformInverse() {
        DCT dct = new DCT().setInverse(true);
        Table outputTable = dct.transform(inputTable)[0];

        verifyTransformResult(
                outputTable, expectedInverseOutputData, dct.getInputCol(), dct.getOutputCol());
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        inputTable = TestUtils.convertDataTypesToSparseInt(tEnv, inputTable);
        assertArrayEquals(
                new Class<?>[] {SparseIntDoubleVector.class},
                TestUtils.getColumnDataTypes(inputTable));

        DCT dct = new DCT();
        Table outputTable = dct.transform(inputTable)[0];

        verifyTransformResult(
                outputTable, expectedForwardOutputData, dct.getInputCol(), dct.getOutputCol());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        DCT dct = new DCT().setInverse(true);

        DCT loadedDCT =
                TestUtils.saveAndReload(
                        tEnv, dct, TEMPORARY_FOLDER.newFolder().getAbsolutePath(), DCT::load);

        Table outputTable = loadedDCT.transform(inputTable)[0];

        verifyTransformResult(
                outputTable,
                expectedInverseOutputData,
                loadedDCT.getInputCol(),
                loadedDCT.getOutputCol());
    }

    @SuppressWarnings("unchecked")
    private static void verifyTransformResult(
            Table outputTable, List<Row> expectedOutputData, String inputCol, String outputCol) {
        List<Row> actualOutputData = IteratorUtils.toList(outputTable.execute().collect());
        actualOutputData.sort(
                Comparator.comparingLong(
                        x ->
                                ((IntDoubleVector) Objects.requireNonNull(x.getField(inputCol)))
                                        .toDense()
                                        .hashCode()));

        expectedOutputData.sort(
                Comparator.comparingLong(
                        x ->
                                ((IntDoubleVector) Objects.requireNonNull(x.getField(0)))
                                        .toDense()
                                        .hashCode()));

        assertEquals(actualOutputData.size(), expectedOutputData.size());
        for (int i = 0; i < actualOutputData.size(); i++) {
            IntDoubleVector actualVector = actualOutputData.get(i).getFieldAs(outputCol);
            IntDoubleVector expectedVector = expectedOutputData.get(i).getFieldAs(1);
            assertArrayEquals(expectedVector.toArray(), actualVector.toArray(), 1e-3);
        }
    }
}
