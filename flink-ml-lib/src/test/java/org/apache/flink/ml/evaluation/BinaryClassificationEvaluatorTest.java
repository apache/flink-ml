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

package org.apache.flink.ml.evaluation;

import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluator;
import org.apache.flink.ml.evaluation.binaryclassification.BinaryClassificationEvaluatorParams;
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
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/** Tests {@link BinaryClassificationEvaluator}. */
public class BinaryClassificationEvaluatorTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table inputDataTable;
    private Table inputDataTableScore;
    private Table inputDataTableWithMultiScore;
    private Table inputDataTableWithWeight;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(1.0, Vectors.dense(0.1, 0.9)),
                    Row.of(1.0, Vectors.dense(0.2, 0.8)),
                    Row.of(1.0, Vectors.dense(0.3, 0.7)),
                    Row.of(0.0, Vectors.dense(0.25, 0.75)),
                    Row.of(0.0, Vectors.dense(0.4, 0.6)),
                    Row.of(1.0, Vectors.dense(0.35, 0.65)),
                    Row.of(1.0, Vectors.dense(0.45, 0.55)),
                    Row.of(0.0, Vectors.dense(0.6, 0.4)),
                    Row.of(0.0, Vectors.dense(0.7, 0.3)),
                    Row.of(1.0, Vectors.dense(0.65, 0.35)),
                    Row.of(0.0, Vectors.dense(0.8, 0.2)),
                    Row.of(1.0, Vectors.dense(0.9, 0.1)));

    private static final List<Row> INPUT_DATA_DOUBLE_RAW =
            Arrays.asList(
                    Row.of(1, 0.9),
                    Row.of(1, 0.8),
                    Row.of(1, 0.7),
                    Row.of(0, 0.75),
                    Row.of(0, 0.6),
                    Row.of(1, 0.65),
                    Row.of(1, 0.55),
                    Row.of(0, 0.4),
                    Row.of(0, 0.3),
                    Row.of(1, 0.35),
                    Row.of(0, 0.2),
                    Row.of(1, 0.1));

    private static final List<Row> INPUT_DATA_WITH_MULTI_SCORE =
            Arrays.asList(
                    Row.of(1.0, Vectors.dense(0.1, 0.9)),
                    Row.of(1.0, Vectors.dense(0.1, 0.9)),
                    Row.of(1.0, Vectors.dense(0.1, 0.9)),
                    Row.of(0.0, Vectors.dense(0.25, 0.75)),
                    Row.of(0.0, Vectors.dense(0.4, 0.6)),
                    Row.of(1.0, Vectors.dense(0.1, 0.9)),
                    Row.of(1.0, Vectors.dense(0.1, 0.9)),
                    Row.of(0.0, Vectors.dense(0.6, 0.4)),
                    Row.of(0.0, Vectors.dense(0.7, 0.3)),
                    Row.of(1.0, Vectors.dense(0.1, 0.9)),
                    Row.of(0.0, Vectors.dense(0.8, 0.2)),
                    Row.of(1.0, Vectors.dense(0.9, 0.1)));

    private static final List<Row> INPUT_DATA_WITH_WEIGHT =
            Arrays.asList(
                    Row.of(1.0, Vectors.dense(0.1, 0.9), 0.8),
                    Row.of(1.0, Vectors.dense(0.1, 0.9), 0.7),
                    Row.of(1.0, Vectors.dense(0.1, 0.9), 0.5),
                    Row.of(0.0, Vectors.dense(0.25, 0.75), 1.2),
                    Row.of(0.0, Vectors.dense(0.4, 0.6), 1.3),
                    Row.of(1.0, Vectors.dense(0.1, 0.9), 1.5),
                    Row.of(1.0, Vectors.dense(0.1, 0.9), 1.4),
                    Row.of(0.0, Vectors.dense(0.6, 0.4), 0.3),
                    Row.of(0.0, Vectors.dense(0.7, 0.3), 0.5),
                    Row.of(1.0, Vectors.dense(0.1, 0.9), 1.9),
                    Row.of(0.0, Vectors.dense(0.8, 0.2), 1.2),
                    Row.of(1.0, Vectors.dense(0.9, 0.1), 1.0));

    private static final double[] EXPECTED_DATA =
            new double[] {0.7691481137909708, 0.3714285714285714, 0.6571428571428571};
    private static final double[] EXPECTED_DATA_M =
            new double[] {
                0.8571428571428571, 0.9377705627705628, 0.8571428571428571, 0.6488095238095237
            };
    private static final double EXPECTED_DATA_W = 0.8911680911680911;
    private static final double EPS = 1.0e-5;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        inputDataTable =
                tEnv.fromDataStream(env.fromCollection(INPUT_DATA)).as("label", "rawPrediction");
        inputDataTableScore =
                tEnv.fromDataStream(env.fromCollection(INPUT_DATA_DOUBLE_RAW))
                        .as("label", "rawPrediction");

        inputDataTableWithMultiScore =
                tEnv.fromDataStream(env.fromCollection(INPUT_DATA_WITH_MULTI_SCORE))
                        .as("label", "rawPrediction");
        inputDataTableWithWeight =
                tEnv.fromDataStream(env.fromCollection(INPUT_DATA_WITH_WEIGHT))
                        .as("label", "rawPrediction", "weight");
    }

    @Test
    public void testParam() {
        BinaryClassificationEvaluator binaryEval = new BinaryClassificationEvaluator();
        assertEquals("label", binaryEval.getLabelCol());
        assertNull(binaryEval.getWeightCol());
        assertEquals("rawPrediction", binaryEval.getRawPredictionCol());
        assertArrayEquals(
                new String[] {
                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR
                },
                binaryEval.getMetricsNames());
        binaryEval
                .setLabelCol("labelCol")
                .setRawPredictionCol("raw")
                .setMetricsNames(BinaryClassificationEvaluatorParams.AREA_UNDER_ROC)
                .setWeightCol("weight");
        assertEquals("labelCol", binaryEval.getLabelCol());
        assertEquals("weight", binaryEval.getWeightCol());
        assertEquals("raw", binaryEval.getRawPredictionCol());
        assertArrayEquals(
                new String[] {BinaryClassificationEvaluatorParams.AREA_UNDER_ROC},
                binaryEval.getMetricsNames());
    }

    @Test
    public void testEvaluate() {
        BinaryClassificationEvaluator eval =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(
                                BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                BinaryClassificationEvaluatorParams.KS,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_ROC);
        Table evalResult = eval.transform(inputDataTable)[0];
        List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
        assertArrayEquals(
                new String[] {
                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                    BinaryClassificationEvaluatorParams.KS,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC
                },
                evalResult.getResolvedSchema().getColumnNames().toArray());
        Row result = results.get(0);
        for (int i = 0; i < EXPECTED_DATA.length; ++i) {
            assertEquals(EXPECTED_DATA[i], result.getFieldAs(i), EPS);
        }
    }

    @Test
    public void testInputTypeConversion() {
        inputDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, inputDataTable);
        assertArrayEquals(
                new Class<?>[] {Integer.class, SparseIntDoubleVector.class},
                TestUtils.getColumnDataTypes(inputDataTable));

        BinaryClassificationEvaluator eval =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(
                                BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                BinaryClassificationEvaluatorParams.KS,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_ROC);
        Table evalResult = eval.transform(inputDataTable)[0];
        List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
        assertArrayEquals(
                new String[] {
                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                    BinaryClassificationEvaluatorParams.KS,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC
                },
                evalResult.getResolvedSchema().getColumnNames().toArray());
        Row result = results.get(0);
        for (int i = 0; i < EXPECTED_DATA.length; ++i) {
            assertEquals(EXPECTED_DATA[i], result.getFieldAs(i), EPS);
        }
    }

    @Test
    public void testEvaluateWithDoubleRaw() {
        BinaryClassificationEvaluator eval =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(
                                BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                BinaryClassificationEvaluatorParams.KS,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_ROC);
        Table evalResult = eval.transform(inputDataTableScore)[0];
        List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
        assertArrayEquals(
                new String[] {
                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                    BinaryClassificationEvaluatorParams.KS,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC
                },
                evalResult.getResolvedSchema().getColumnNames().toArray());
        Row result = results.get(0);
        for (int i = 0; i < EXPECTED_DATA.length; ++i) {
            assertEquals(EXPECTED_DATA[i], result.getFieldAs(i), EPS);
        }
    }

    @Test
    public void testMoreSubtaskThanData() {
        List<Row> inputData =
                Arrays.asList(
                        Row.of(1.0, Vectors.dense(0.1, 0.9)), Row.of(0.0, Vectors.dense(0.9, 0.1)));
        double[] expectedData = new double[] {1.0, 1.0, 1.0};
        inputDataTable =
                tEnv.fromDataStream(env.fromCollection(inputData)).as("label", "rawPrediction");

        BinaryClassificationEvaluator eval =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(
                                BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                BinaryClassificationEvaluatorParams.KS,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_ROC);
        Table evalResult = eval.transform(inputDataTable)[0];
        List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
        assertArrayEquals(
                new String[] {
                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                    BinaryClassificationEvaluatorParams.KS,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC
                },
                evalResult.getResolvedSchema().getColumnNames().toArray());
        Row result = results.get(0);
        for (int i = 0; i < expectedData.length; ++i) {
            assertEquals(expectedData[i], result.getFieldAs(i), EPS);
        }
    }

    @Test
    public void testEvaluateWithMultiScore() {
        BinaryClassificationEvaluator eval =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(
                                BinaryClassificationEvaluatorParams.AREA_UNDER_ROC,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                BinaryClassificationEvaluatorParams.KS,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_LORENZ);
        Table evalResult = eval.transform(inputDataTableWithMultiScore)[0];
        List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
        assertArrayEquals(
                new String[] {
                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                    BinaryClassificationEvaluatorParams.KS,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_LORENZ
                },
                evalResult.getResolvedSchema().getColumnNames().toArray());
        Row result = results.get(0);
        for (int i = 0; i < EXPECTED_DATA_M.length; ++i) {
            assertEquals(EXPECTED_DATA_M[i], result.getFieldAs(i), EPS);
        }
    }

    @Test
    public void testEvaluateWithWeight() {
        BinaryClassificationEvaluator eval =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(BinaryClassificationEvaluatorParams.AREA_UNDER_ROC)
                        .setWeightCol("weight");
        Table evalResult = eval.transform(inputDataTableWithWeight)[0];
        List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
        assertArrayEquals(
                new String[] {BinaryClassificationEvaluatorParams.AREA_UNDER_ROC},
                evalResult.getResolvedSchema().getColumnNames().toArray());
        assertEquals(EXPECTED_DATA_W, results.get(0).getFieldAs(0), EPS);
    }

    @Test
    public void testSaveLoadAndEvaluate() throws Exception {
        BinaryClassificationEvaluator eval =
                new BinaryClassificationEvaluator()
                        .setMetricsNames(
                                BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                                BinaryClassificationEvaluatorParams.KS,
                                BinaryClassificationEvaluatorParams.AREA_UNDER_ROC);
        BinaryClassificationEvaluator loadedEval =
                TestUtils.saveAndReload(
                        tEnv,
                        eval,
                        tempFolder.newFolder().getAbsolutePath(),
                        BinaryClassificationEvaluator::load);
        Table evalResult = loadedEval.transform(inputDataTable)[0];
        assertArrayEquals(
                new String[] {
                    BinaryClassificationEvaluatorParams.AREA_UNDER_PR,
                    BinaryClassificationEvaluatorParams.KS,
                    BinaryClassificationEvaluatorParams.AREA_UNDER_ROC
                },
                evalResult.getResolvedSchema().getColumnNames().toArray());
        List<Row> results = IteratorUtils.toList(evalResult.execute().collect());
        Row result = results.get(0);
        for (int i = 0; i < EXPECTED_DATA.length; ++i) {
            assertEquals(EXPECTED_DATA[i], result.getFieldAs(i), EPS);
        }
    }
}
