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

import org.apache.flink.ml.feature.polynomialexpansion.PolynomialExpansion;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link PolynomialExpansion}. */
public class PolynomialExpansionTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            Vectors.dense(1.0, 2.0, 3.0),
                            Vectors.sparse(5, new int[] {1, 4}, new double[] {2.0, 3.0})),
                    Row.of(
                            Vectors.dense(2.0, 3.0),
                            Vectors.sparse(5, new int[] {1, 4}, new double[] {2.0, 1.0})));

    private static final List<Vector> EXPECTED_DENSE_OUTPUT =
            Arrays.asList(
                    Vectors.dense(1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 3.0, 6.0, 9.0),
                    Vectors.dense(2.0, 4.0, 3.0, 6.0, 9.0));

    private static final List<Vector> EXPECTED_DENSE_OUTPUT_WITH_DEGREE_3 =
            Arrays.asList(
                    Vectors.dense(
                            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 8.0, 3.0, 3.0, 3.0, 6.0, 6.0,
                            12.0, 9.0, 9.0, 18.0, 27.0),
                    Vectors.dense(2.0, 4.0, 8.0, 3.0, 6.0, 12.0, 9.0, 18.0, 27.0));

    private static final List<Vector> EXPECTED_SPARSE_OUTPUT =
            Arrays.asList(
                    Vectors.sparse(
                            55,
                            new int[] {3, 6, 8, 34, 37, 39, 49, 51, 54},
                            new double[] {2.0, 4.0, 8.0, 3.0, 6.0, 12.0, 9.0, 18.0, 27.0}),
                    Vectors.sparse(
                            55,
                            new int[] {3, 6, 8, 34, 37, 39, 49, 51, 54},
                            new double[] {2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 1.0, 2.0, 1.0}));

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT_DATA);
        inputDataTable = tEnv.fromDataStream(dataStream).as("denseVec", "sparseVec");
    }

    private void verifyOutputResult(Table output, String outputCol, List<Vector> expectedData)
            throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Row> stream = tEnv.toDataStream(output);

        List<Row> results = IteratorUtils.toList(stream.executeAndCollect());
        List<Vector> resultVec = new ArrayList<>(results.size());
        for (Row row : results) {
            if (row.getField(outputCol) != null) {
                resultVec.add(row.getFieldAs(outputCol));
            }
        }
        compareResultCollections(expectedData, resultVec, TestUtils::compare);
    }

    @Test
    public void testParam() {
        PolynomialExpansion polynomialExpansion = new PolynomialExpansion();
        assertEquals("input", polynomialExpansion.getInputCol());
        assertEquals("output", polynomialExpansion.getOutputCol());
        assertEquals(2, polynomialExpansion.getDegree());

        polynomialExpansion.setInputCol("denseVec").setOutputCol("outputVec").setDegree(5);
        assertEquals("denseVec", polynomialExpansion.getInputCol());
        assertEquals("outputVec", polynomialExpansion.getOutputCol());
        assertEquals(5, polynomialExpansion.getDegree());
    }

    @Test
    public void testOutputSchema() {
        PolynomialExpansion polynomialExpansion =
                new PolynomialExpansion()
                        .setInputCol("denseVec")
                        .setOutputCol("outputVec")
                        .setDegree(3);

        Table output = polynomialExpansion.transform(inputDataTable)[0];

        assertEquals(
                Arrays.asList("denseVec", "sparseVec", "outputVec"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        PolynomialExpansion polynomialExpansion =
                new PolynomialExpansion()
                        .setInputCol("denseVec")
                        .setOutputCol("outputVec")
                        .setDegree(2);

        PolynomialExpansion loadedPolynomialExpansion =
                TestUtils.saveAndReload(
                        tEnv,
                        polynomialExpansion,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        PolynomialExpansion::load);

        Table output = loadedPolynomialExpansion.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedPolynomialExpansion.getOutputCol(), EXPECTED_DENSE_OUTPUT);
    }

    @Test
    public void testInvalidDegree() {
        try {
            PolynomialExpansion polynomialExpansion =
                    new PolynomialExpansion()
                            .setInputCol("denseVec")
                            .setOutputCol("outputVec")
                            .setDegree(-1);
            polynomialExpansion.transform(inputDataTable);
            fail();
        } catch (Exception e) {
            assertEquals("Parameter degree is given an invalid value -1", e.getMessage());
        }
    }

    @Test
    public void testDenseTransform() throws Exception {
        PolynomialExpansion polynomialExpansion =
                new PolynomialExpansion()
                        .setInputCol("denseVec")
                        .setOutputCol("outputVec")
                        .setDegree(3);

        Table output = polynomialExpansion.transform(inputDataTable)[0];
        verifyOutputResult(
                output, polynomialExpansion.getOutputCol(), EXPECTED_DENSE_OUTPUT_WITH_DEGREE_3);
    }

    @Test
    public void testSparseTransform() throws Exception {
        PolynomialExpansion polynomialExpansion =
                new PolynomialExpansion()
                        .setInputCol("sparseVec")
                        .setOutputCol("outputVec")
                        .setDegree(3);

        Table output = polynomialExpansion.transform(inputDataTable)[0];
        verifyOutputResult(output, polynomialExpansion.getOutputCol(), EXPECTED_SPARSE_OUTPUT);
    }
}
