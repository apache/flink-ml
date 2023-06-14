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

import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizer;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerModel;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerModelData;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerParams;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.apache.flink.table.api.Expressions.$;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link KBinsDiscretizer} and {@link KBinsDiscretizerModel}. */
public class KBinsDiscretizerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainTable;
    private Table testTable;

    // Column0 for normal cases, column1 for constant cases, column2 for numDistinct < numBins
    // cases.
    private static final List<Row> TRAIN_INPUT =
            Arrays.asList(
                    Row.of(Vectors.dense(1, 10, 0)),
                    Row.of(Vectors.dense(1, 10, 0)),
                    Row.of(Vectors.dense(1, 10, 0)),
                    Row.of(Vectors.dense(4, 10, 0)),
                    Row.of(Vectors.dense(5, 10, 0)),
                    Row.of(Vectors.dense(6, 10, 0)),
                    Row.of(Vectors.dense(7, 10, 0)),
                    Row.of(Vectors.dense(10, 10, 0)),
                    Row.of(Vectors.dense(13, 10, 3)));

    private static final List<Row> TEST_INPUT =
            Arrays.asList(
                    Row.of(Vectors.dense(-1, 0, 0)),
                    Row.of(Vectors.dense(1, 1, 1)),
                    Row.of(Vectors.dense(1.5, 1, 2)),
                    Row.of(Vectors.dense(5, 2, 3)),
                    Row.of(Vectors.dense(7.25, 3, 4)),
                    Row.of(Vectors.dense(13, 4, 5)),
                    Row.of(Vectors.dense(15, 4, 6)));

    private static final double[][] UNIFORM_MODEL_DATA =
            new double[][] {
                new double[] {1, 5, 9, 13},
                new double[] {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY},
                new double[] {0, 1, 2, 3}
            };

    private static final List<Row> UNIFORM_OUTPUT =
            Arrays.asList(
                    Row.of(Vectors.dense(0, 0, 0)),
                    Row.of(Vectors.dense(0, 0, 1)),
                    Row.of(Vectors.dense(0, 0, 2)),
                    Row.of(Vectors.dense(1, 0, 2)),
                    Row.of(Vectors.dense(1, 0, 2)),
                    Row.of(Vectors.dense(2, 0, 2)),
                    Row.of(Vectors.dense(2, 0, 2)));

    private static final List<Row> QUANTILE_OUTPUT =
            Arrays.asList(
                    Row.of(Vectors.dense(0, 0, 0)),
                    Row.of(Vectors.dense(0, 0, 0)),
                    Row.of(Vectors.dense(0, 0, 1)),
                    Row.of(Vectors.dense(1, 0, 1)),
                    Row.of(Vectors.dense(2, 0, 1)),
                    Row.of(Vectors.dense(2, 0, 1)),
                    Row.of(Vectors.dense(2, 0, 1)));

    private static final List<Row> KMEANS_OUTPUT =
            Arrays.asList(
                    Row.of(Vectors.dense(0, 0, 0)),
                    Row.of(Vectors.dense(0, 0, 1)),
                    Row.of(Vectors.dense(0, 0, 2)),
                    Row.of(Vectors.dense(1, 0, 2)),
                    Row.of(Vectors.dense(1, 0, 2)),
                    Row.of(Vectors.dense(2, 0, 2)),
                    Row.of(Vectors.dense(2, 0, 2)));

    private static final double TOLERANCE = 1e-7;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        trainTable = tEnv.fromDataStream(env.fromCollection(TRAIN_INPUT)).as("input");
        testTable = tEnv.fromDataStream(env.fromCollection(TEST_INPUT)).as("input");
    }

    @SuppressWarnings("unchecked, ConstantConditions")
    private void verifyPredictionResult(
            List<Row> expectedOutput, Table output, String predictionCol) throws Exception {
        List<Row> collectedResult =
                IteratorUtils.toList(
                        tEnv.toDataStream(output.select($(predictionCol))).executeAndCollect());
        TestBaseUtils.compareResultCollections(
                expectedOutput,
                collectedResult,
                (o1, o2) ->
                        TestUtils.compare(
                                (DenseIntDoubleVector) o1.getField(0),
                                (DenseIntDoubleVector) o2.getField(0)));
    }

    @Test
    public void testParam() {
        KBinsDiscretizer kBinsDiscretizer = new KBinsDiscretizer();

        assertEquals("input", kBinsDiscretizer.getInputCol());
        assertEquals(5, kBinsDiscretizer.getNumBins());
        assertEquals("quantile", kBinsDiscretizer.getStrategy());
        assertEquals(200000, kBinsDiscretizer.getSubSamples());
        assertEquals("output", kBinsDiscretizer.getOutputCol());

        kBinsDiscretizer
                .setInputCol("test_input")
                .setNumBins(10)
                .setStrategy(KBinsDiscretizerParams.KMEANS)
                .setSubSamples(1000)
                .setOutputCol("test_output");

        assertEquals("test_input", kBinsDiscretizer.getInputCol());
        assertEquals(10, kBinsDiscretizer.getNumBins());
        assertEquals("kmeans", kBinsDiscretizer.getStrategy());
        assertEquals(1000, kBinsDiscretizer.getSubSamples());
        assertEquals("test_output", kBinsDiscretizer.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable =
                tEnv.fromDataStream(env.fromElements(Row.of("", "")))
                        .as("test_input", "dummy_input");
        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer().setInputCol("test_input").setOutputCol("test_output");
        Table output = kBinsDiscretizer.fit(tempTable).transform(tempTable)[0];

        assertEquals(
                Arrays.asList("test_input", "dummy_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        KBinsDiscretizer kBinsDiscretizer = new KBinsDiscretizer().setNumBins(3);
        Table output;

        // Tests uniform strategy.
        kBinsDiscretizer.setStrategy(KBinsDiscretizerParams.UNIFORM);
        output = kBinsDiscretizer.fit(trainTable).transform(testTable)[0];
        verifyPredictionResult(UNIFORM_OUTPUT, output, kBinsDiscretizer.getOutputCol());

        // Tests quantile strategy.
        kBinsDiscretizer.setStrategy(KBinsDiscretizerParams.QUANTILE);
        output = kBinsDiscretizer.fit(trainTable).transform(testTable)[0];
        verifyPredictionResult(QUANTILE_OUTPUT, output, kBinsDiscretizer.getOutputCol());

        // Tests kmeans strategy.
        kBinsDiscretizer.setStrategy(KBinsDiscretizerParams.KMEANS);
        output = kBinsDiscretizer.fit(trainTable).transform(testTable)[0];
        verifyPredictionResult(KMEANS_OUTPUT, output, kBinsDiscretizer.getOutputCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer().setNumBins(3).setStrategy(KBinsDiscretizerParams.UNIFORM);
        kBinsDiscretizer =
                TestUtils.saveAndReload(
                        tEnv,
                        kBinsDiscretizer,
                        tempFolder.newFolder().getAbsolutePath(),
                        KBinsDiscretizer::load);

        KBinsDiscretizerModel model = kBinsDiscretizer.fit(trainTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        KBinsDiscretizerModel::load);

        assertEquals(
                Collections.singletonList("binEdges"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        Table output = model.transform(testTable)[0];
        verifyPredictionResult(UNIFORM_OUTPUT, output, kBinsDiscretizer.getOutputCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer().setNumBins(3).setStrategy(KBinsDiscretizerParams.UNIFORM);
        KBinsDiscretizerModel model = kBinsDiscretizer.fit(trainTable);
        Table modelDataTable = model.getModelData()[0];

        assertEquals(
                Collections.singletonList("binEdges"),
                modelDataTable.getResolvedSchema().getColumnNames());

        List<KBinsDiscretizerModelData> collectedModelData =
                (List<KBinsDiscretizerModelData>)
                        IteratorUtils.toList(
                                KBinsDiscretizerModelData.getModelDataStream(modelDataTable)
                                        .executeAndCollect());
        assertEquals(1, collectedModelData.size());

        KBinsDiscretizerModelData modelData = collectedModelData.get(0);
        assertEquals(UNIFORM_MODEL_DATA.length, modelData.binEdges.length);
        for (int i = 0; i < modelData.binEdges.length; i++) {
            assertArrayEquals(UNIFORM_MODEL_DATA[i], modelData.binEdges[i], TOLERANCE);
        }
    }

    @Test
    public void testSetModelData() throws Exception {
        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer().setNumBins(3).setStrategy(KBinsDiscretizerParams.UNIFORM);

        KBinsDiscretizerModel model = kBinsDiscretizer.fit(trainTable);

        KBinsDiscretizerModel newModel = new KBinsDiscretizerModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(testTable)[0];

        verifyPredictionResult(UNIFORM_OUTPUT, output, kBinsDiscretizer.getOutputCol());
    }

    @Test
    public void testFitOnEmptyData() {
        Table emptyTable =
                tEnv.fromDataStream(env.fromCollection(TRAIN_INPUT).filter(x -> x.getArity() == 0))
                        .as("input");
        KBinsDiscretizerModel model = new KBinsDiscretizer().fit(emptyTable);
        Table modelDataTable = model.getModelData()[0];
        try {
            modelDataTable.execute().collect().next();
            fail();
        } catch (Throwable e) {
            assertEquals("The training set is empty.", ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testBinsWithWidthAsZero() throws Exception {
        final List<Row> expectedOutput =
                Arrays.asList(
                        Row.of(Vectors.dense(0, 0, 0)),
                        Row.of(Vectors.dense(0, 0, 0)),
                        Row.of(Vectors.dense(0, 0, 1)),
                        Row.of(Vectors.dense(3, 0, 1)),
                        Row.of(Vectors.dense(5, 0, 1)),
                        Row.of(Vectors.dense(6, 0, 1)),
                        Row.of(Vectors.dense(6, 0, 1)));

        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer().setNumBins(10).setStrategy(KBinsDiscretizerParams.QUANTILE);

        Table output = kBinsDiscretizer.fit(trainTable).transform(testTable)[0];
        verifyPredictionResult(expectedOutput, output, kBinsDiscretizer.getOutputCol());
    }
}
