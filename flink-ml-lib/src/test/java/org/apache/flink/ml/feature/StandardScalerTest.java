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
import org.apache.flink.ml.feature.standardscaler.StandardScaler;
import org.apache.flink.ml.feature.standardscaler.StandardScalerModel;
import org.apache.flink.ml.feature.standardscaler.StandardScalerModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link StandardScaler} and {@link StandardScalerModel}. */
public class StandardScalerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table denseTable;

    private final List<Row> denseInput =
            Arrays.asList(
                    Row.of(Vectors.dense(-2.5, 9, 1)),
                    Row.of(Vectors.dense(1.4, -5, 1)),
                    Row.of(Vectors.dense(2, -1, -2)));

    private final List<DenseVector> expectedResWithMean =
            Arrays.asList(
                    Vectors.dense(-2.8, 8, 1),
                    Vectors.dense(1.1, -6, 1),
                    Vectors.dense(1.7, -2, -2));

    private final List<DenseVector> expectedResWithStd =
            Arrays.asList(
                    Vectors.dense(-1.0231819, 1.2480754, 0.5773502),
                    Vectors.dense(0.5729819, -0.6933752, 0.5773503),
                    Vectors.dense(0.8185455, -0.1386750, -1.1547005));

    private final List<DenseVector> expectedResWithMeanAndStd =
            Arrays.asList(
                    Vectors.dense(-1.1459637, 1.1094004, 0.5773503),
                    Vectors.dense(0.45020003, -0.8320503, 0.5773503),
                    Vectors.dense(0.69576368, -0.2773501, -1.1547005));

    private final double[] expectedMean = new double[] {0.3, 1, 0};
    private final double[] expectedStd = new double[] {2.4433583, 7.2111026, 1.7320508};
    private static final double TOLERANCE = 1e-7;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        denseTable = tEnv.fromDataStream(env.fromCollection(denseInput)).as("input");
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(
            List<DenseVector> expectedOutput, Table output, String predictionCol) throws Exception {
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<DenseVector> predictions = new ArrayList<>(collectedResult.size());

        for (Row r : collectedResult) {
            Vector vec = (Vector) r.getField(predictionCol);
            predictions.add(vec.toDense());
        }

        assertEquals(expectedOutput.size(), predictions.size());

        predictions.sort(
                (vec1, vec2) -> {
                    int size = Math.min(vec1.size(), vec2.size());
                    for (int i = 0; i < size; i++) {
                        int cmp = Double.compare(vec1.get(i), vec2.get(i));
                        if (cmp != 0) {
                            return cmp;
                        }
                    }
                    return 0;
                });

        for (int i = 0; i < predictions.size(); i++) {
            assertArrayEquals(expectedOutput.get(i).values, predictions.get(i).values, TOLERANCE);
        }
    }

    @Test
    public void testParam() {
        StandardScaler standardScaler = new StandardScaler();

        assertEquals("input", standardScaler.getInputCol());
        assertEquals(false, standardScaler.getWithMean());
        assertEquals(true, standardScaler.getWithStd());
        assertEquals("output", standardScaler.getOutputCol());

        standardScaler
                .setInputCol("test_input")
                .setWithMean(true)
                .setWithStd(false)
                .setOutputCol("test_output");

        assertEquals("test_input", standardScaler.getInputCol());
        assertEquals(true, standardScaler.getWithMean());
        assertEquals(false, standardScaler.getWithStd());
        assertEquals("test_output", standardScaler.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = denseTable.as("test_input");
        StandardScaler standardScaler =
                new StandardScaler().setInputCol("test_input").setOutputCol("test_output");
        Table output = standardScaler.fit(tempTable).transform(tempTable)[0];

        assertEquals(
                Arrays.asList("test_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredictWithStd() throws Exception {
        StandardScaler standardScaler = new StandardScaler();
        Table output = standardScaler.fit(denseTable).transform(denseTable)[0];
        verifyPredictionResult(expectedResWithStd, output, standardScaler.getOutputCol());
    }

    @Test
    public void testFitAndPredictWithMean() throws Exception {
        StandardScaler standardScaler = new StandardScaler().setWithStd(false).setWithMean(true);
        Table output = standardScaler.fit(denseTable).transform(denseTable)[0];
        verifyPredictionResult(expectedResWithMean, output, standardScaler.getOutputCol());
    }

    @Test
    public void testFitAndPredictWithMeanAndStd() throws Exception {
        StandardScaler standardScaler = new StandardScaler().setWithMean(true);
        Table output = standardScaler.fit(denseTable).transform(denseTable)[0];
        verifyPredictionResult(expectedResWithMeanAndStd, output, standardScaler.getOutputCol());
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        denseTable = TestUtils.convertDataTypesToSparseInt(tEnv, denseTable);
        assertArrayEquals(
                new Class<?>[] {SparseVector.class}, TestUtils.getColumnDataTypes(denseTable));

        StandardScaler standardScaler = new StandardScaler().setWithMean(true);
        Table output = standardScaler.fit(denseTable).transform(denseTable)[0];
        verifyPredictionResult(expectedResWithMeanAndStd, output, standardScaler.getOutputCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        StandardScaler standardScaler = new StandardScaler();
        standardScaler =
                TestUtils.saveAndReload(
                        tEnv, standardScaler, tempFolder.newFolder().getAbsolutePath());

        StandardScalerModel model = standardScaler.fit(denseTable);
        model = TestUtils.saveAndReload(tEnv, model, tempFolder.newFolder().getAbsolutePath());

        assertEquals(
                Arrays.asList("mean", "std"),
                model.getModelData()[0].getResolvedSchema().getColumnNames().subList(0, 2));

        Table output = model.transform(denseTable)[0];
        verifyPredictionResult(expectedResWithStd, output, standardScaler.getOutputCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        StandardScaler standardScaler = new StandardScaler();
        StandardScalerModel model = standardScaler.fit(denseTable);
        Table modelDataTable = model.getModelData()[0];

        assertEquals(
                Arrays.asList("mean", "std"),
                modelDataTable.getResolvedSchema().getColumnNames().subList(0, 2));

        List<StandardScalerModelData> collectedModelData =
                (List<StandardScalerModelData>)
                        IteratorUtils.toList(
                                StandardScalerModelData.getModelDataStream(modelDataTable)
                                        .executeAndCollect());
        assertEquals(1, collectedModelData.size());

        StandardScalerModelData modelData = collectedModelData.get(0);
        assertArrayEquals(expectedMean, modelData.mean.values, TOLERANCE);
        assertArrayEquals(expectedStd, modelData.std.values, TOLERANCE);
    }

    @Test
    public void testSetModelData() throws Exception {
        StandardScaler standardScaler = new StandardScaler();
        StandardScalerModel model = standardScaler.fit(denseTable);

        StandardScalerModel newModel = new StandardScalerModel();
        ReadWriteUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(denseTable)[0];

        verifyPredictionResult(expectedResWithStd, output, standardScaler.getOutputCol());
    }

    @Test
    public void testSparseInput() throws Exception {
        final List<Row> sparseInput =
                Arrays.asList(
                        Row.of(Vectors.sparse(3, new int[] {0, 1}, new double[] {-2.5, 1})),
                        Row.of(Vectors.sparse(3, new int[] {1, 2}, new double[] {2, -2})),
                        Row.of(Vectors.sparse(3, new int[] {0, 2}, new double[] {1.4, 1})));
        Table sparseTable = tEnv.fromDataStream(env.fromCollection(sparseInput)).as("input");

        final List<DenseVector> expectedResWithStd =
                Arrays.asList(
                        Vectors.dense(-1.2653836, 1, 0),
                        Vectors.dense(0, 2, -1.30930734),
                        Vectors.dense(0.7086148, 0, 0.6546537));
        StandardScaler standardScaler = new StandardScaler();
        Table output = standardScaler.fit(sparseTable).transform(sparseTable)[0];

        verifyPredictionResult(expectedResWithStd, output, standardScaler.getOutputCol());
    }

    @Test
    public void testFitOnEmptyData() {
        Table emptyTable =
                tEnv.fromDataStream(env.fromCollection(denseInput).filter(x -> x.getArity() == 0))
                        .as("input");
        StandardScalerModel model = new StandardScaler().fit(emptyTable);
        Table modelDataTable = model.getModelData()[0];
        try {
            IteratorUtils.toList(
                    StandardScalerModelData.getModelDataStream(modelDataTable).executeAndCollect());
            fail();
        } catch (Throwable e) {
            assertEquals("The training set is empty.", ExceptionUtils.getRootCause(e).getMessage());
        }
    }
}
