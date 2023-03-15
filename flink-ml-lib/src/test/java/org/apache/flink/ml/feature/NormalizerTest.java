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

import org.apache.flink.ml.feature.normalizer.Normalizer;
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

/** Tests {@link Normalizer}. */
public class NormalizerTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            Vectors.dense(2.1, 3.1, 2.3, 3.4, 5.3, 5.1),
                            Vectors.sparse(5, new int[] {1, 3, 4}, new double[] {0.1, 0.2, 0.3})),
                    Row.of(
                            Vectors.dense(2.3, 4.1, 1.3, 2.4, 5.1, 4.1),
                            Vectors.sparse(5, new int[] {1, 2, 4}, new double[] {0.1, 0.2, 0.3})));

    private static final List<Vector> EXPECTED_DENSE_OUTPUT =
            Arrays.asList(
                    Vectors.dense(
                            0.17386300895299714,
                            0.25665491797823387,
                            0.19042139075804446,
                            0.28149249068580484,
                            0.43879711783375464,
                            0.42223873602870726),
                    Vectors.dense(
                            0.20785190042726007,
                            0.3705186051094636,
                            0.11748150893714701,
                            0.2168889395762714,
                            0.4608889965995767,
                            0.3705186051094636));

    private static final List<Vector> EXPECTED_SPARSE_OUTPUT =
            Arrays.asList(
                    Vectors.sparse(
                            5,
                            new int[] {1, 3, 4},
                            new double[] {
                                0.23070057753660791, 0.46140115507321583, 0.6921017326098237
                            }),
                    Vectors.sparse(
                            5,
                            new int[] {1, 2, 4},
                            new double[] {
                                0.23070057753660791, 0.46140115507321583, 0.6921017326098237
                            }));

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
        Normalizer normalizer = new Normalizer();
        assertEquals("input", normalizer.getInputCol());
        assertEquals("output", normalizer.getOutputCol());
        assertEquals(2.0, normalizer.getP(), 1.0e-5);

        normalizer.setInputCol("denseVec").setOutputCol("outputVec").setP(1.5);
        assertEquals("denseVec", normalizer.getInputCol());
        assertEquals("outputVec", normalizer.getOutputCol());
        assertEquals(1.5, normalizer.getP(), 1.0e-5);
    }

    @Test
    public void testOutputSchema() {
        Normalizer normalizer =
                new Normalizer().setInputCol("denseVec").setOutputCol("outputVec").setP(1.5);

        Table output = normalizer.transform(inputDataTable)[0];

        assertEquals(
                Arrays.asList("denseVec", "sparseVec", "outputVec"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        Normalizer normalizer =
                new Normalizer().setInputCol("denseVec").setOutputCol("outputVec").setP(1.5);

        Normalizer loadedNormalizer =
                TestUtils.saveAndReload(
                        tEnv,
                        normalizer,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        Normalizer::load);

        Table output = loadedNormalizer.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedNormalizer.getOutputCol(), EXPECTED_DENSE_OUTPUT);
    }

    @Test
    public void testInvalidP() {
        try {
            Normalizer normalizer =
                    new Normalizer().setInputCol("denseVec").setOutputCol("outputVec").setP(0.5);
            normalizer.transform(inputDataTable);
            fail();
        } catch (Exception e) {
            assertEquals("Parameter p is given an invalid value 0.5", e.getMessage());
        }
    }

    @Test
    public void testDenseTransform() throws Exception {
        Normalizer normalizer =
                new Normalizer().setInputCol("denseVec").setOutputCol("outputVec").setP(1.5);

        Table output = normalizer.transform(inputDataTable)[0];
        verifyOutputResult(output, normalizer.getOutputCol(), EXPECTED_DENSE_OUTPUT);
    }

    @Test
    public void testSparseTransform() throws Exception {
        Normalizer normalizer =
                new Normalizer().setInputCol("sparseVec").setOutputCol("outputVec").setP(1.5);

        Table output = normalizer.transform(inputDataTable)[0];
        verifyOutputResult(output, normalizer.getOutputCol(), EXPECTED_SPARSE_OUTPUT);
    }
}
