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

package org.apache.flink.ml.classification;

import org.apache.flink.ml.classification.naivebayes.NaiveBayes;
import org.apache.flink.ml.classification.naivebayes.NaiveBayesModel;
import org.apache.flink.ml.classification.naivebayes.NaiveBayesModelData;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link NaiveBayes} and {@link NaiveBayesModel}. */
public class NaiveBayesTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainTable;
    private Table predictTable;
    private Map<IntDoubleVector, Double> expectedOutput;
    private NaiveBayes estimator;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        List<Row> trainData =
                Arrays.asList(
                        Row.of(Vectors.dense(0, 0.), 11),
                        Row.of(Vectors.dense(1, 0), 10),
                        Row.of(Vectors.dense(1, 1.), 10));

        trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("features", "label");

        List<Row> predictData =
                Arrays.asList(
                        Row.of(Vectors.dense(0, 1.)),
                        Row.of(Vectors.dense(0, 0.)),
                        Row.of(Vectors.dense(1, 0)),
                        Row.of(Vectors.dense(1, 1.)));

        predictTable = tEnv.fromDataStream(env.fromCollection(predictData)).as("features");

        expectedOutput =
                new HashMap<IntDoubleVector, Double>() {
                    {
                        put(Vectors.dense(0, 1.), 11.0);
                        put(Vectors.dense(0, 0.), 11.0);
                        put(Vectors.dense(1, 0.), 10.0);
                        put(Vectors.dense(1, 1.), 10.0);
                    }
                };

        estimator =
                new NaiveBayes()
                        .setSmoothing(1.0)
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setModelType("multinomial");
    }

    /**
     * Executes a given table and collect its results. Results are returned as a map whose key is
     * the feature, value is the prediction result.
     *
     * @param table A table to be executed and to have its result collected
     * @param featuresCol Name of the column in the table that contains the features
     * @param predictionCol Name of the column in the table that contains the prediction result
     * @return A map containing the collected results
     */
    private static Map<IntDoubleVector, Double> executeAndCollect(
            Table table, String featuresCol, String predictionCol) {
        Map<IntDoubleVector, Double> map = new HashMap<>();
        for (CloseableIterator<Row> it = table.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            map.put(
                    ((IntDoubleVector) row.getField(featuresCol)).toDense(),
                    (Double) row.getField(predictionCol));
        }

        return map;
    }

    @Test
    public void testParam() {
        NaiveBayes estimator = new NaiveBayes();

        assertEquals("features", estimator.getFeaturesCol());
        assertEquals("label", estimator.getLabelCol());
        assertEquals("multinomial", estimator.getModelType());
        assertEquals("prediction", estimator.getPredictionCol());
        assertEquals(1.0, estimator.getSmoothing(), 1e-5);

        estimator
                .setFeaturesCol("test_feature")
                .setLabelCol("test_label")
                .setPredictionCol("test_prediction")
                .setSmoothing(2.0);

        assertEquals("test_feature", estimator.getFeaturesCol());
        assertEquals("test_label", estimator.getLabelCol());
        assertEquals("test_prediction", estimator.getPredictionCol());
        assertEquals(2.0, estimator.getSmoothing(), 1e-5);

        NaiveBayesModel model = new NaiveBayesModel();

        assertEquals("features", model.getFeaturesCol());
        assertEquals("multinomial", model.getModelType());
        assertEquals("prediction", model.getPredictionCol());

        model.setFeaturesCol("test_feature").setPredictionCol("test_prediction");

        assertEquals("test_feature", model.getFeaturesCol());
        assertEquals("test_prediction", model.getPredictionCol());
    }

    @Test
    public void testFitAndPredict() {
        NaiveBayesModel model = estimator.fit(trainTable);
        Table outputTable = model.transform(predictTable)[0];
        Map<IntDoubleVector, Double> actualOutput =
                executeAndCollect(outputTable, model.getFeaturesCol(), model.getPredictionCol());
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testInputTypeConversion() {
        trainTable = TestUtils.convertDataTypesToSparseInt(tEnv, trainTable);
        predictTable = TestUtils.convertDataTypesToSparseInt(tEnv, predictTable);

        assertArrayEquals(
                new Class<?>[] {SparseIntDoubleVector.class, Integer.class},
                TestUtils.getColumnDataTypes(trainTable));
        assertArrayEquals(
                new Class<?>[] {SparseIntDoubleVector.class},
                TestUtils.getColumnDataTypes(predictTable));

        NaiveBayesModel model = estimator.fit(trainTable);
        Table outputTable = model.transform(predictTable)[0];
        Map<IntDoubleVector, Double> actualOutput =
                executeAndCollect(outputTable, model.getFeaturesCol(), model.getPredictionCol());
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testOutputSchema() {
        trainTable = trainTable.as("test_features", "test_label");
        predictTable = predictTable.as("test_features");

        estimator
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setPredictionCol("test_prediction");

        NaiveBayesModel model = estimator.fit(trainTable);
        Table outputTable = model.transform(predictTable)[0];
        Map<IntDoubleVector, Double> actualOutput =
                executeAndCollect(outputTable, model.getFeaturesCol(), model.getPredictionCol());
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testPredictUnseenFeature() {
        predictTable =
                tEnv.fromDataStream(env.fromElements(Row.of(Vectors.dense(2, 1.)))).as("features");

        NaiveBayesModel model = estimator.fit(trainTable);
        Table outputTable = model.transform(predictTable)[0];

        try {
            outputTable.execute().collect().next();
            Assert.fail("Expected NullPointerException");
        } catch (Exception e) {
            Throwable exception = ExceptionUtils.getRootCause(e);
            assertEquals(
                    NaiveBayesModel.class.getName(), exception.getStackTrace()[0].getClassName());
            assertEquals("calculateProb", exception.getStackTrace()[0].getMethodName());
            assertEquals(NullPointerException.class, exception.getClass());
        }
    }

    @Test
    public void testVectorWithDiffLen() {
        List<Row> trainData =
                Arrays.asList(
                        Row.of(Vectors.dense(0, 0.), 11.0),
                        Row.of(Vectors.dense(1, 0), 10.0),
                        Row.of(Vectors.dense(1), 10.0));

        trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("features", "label");

        NaiveBayesModel model = estimator.fit(trainTable);
        Table outputTable = model.transform(trainTable)[0];

        try {
            outputTable.execute().collect().next();
            Assert.fail("Expected IllegalArgumentException");
        } catch (Exception e) {
            Throwable exception = ExceptionUtils.getRootCause(e);
            assertEquals(IllegalArgumentException.class, exception.getClass());
            assertEquals("Feature vectors should be of equal length.", exception.getMessage());
        }
    }

    @Test
    public void testVectorWithDiffLen2() {
        List<Row> trainData =
                Arrays.asList(Row.of(Vectors.dense(0, 0.), 11.0), Row.of(Vectors.dense(1), 10.0));

        trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("features", "label");

        NaiveBayesModel model = estimator.fit(trainTable);
        Table outputTable = model.transform(trainTable)[0];

        try {
            outputTable.execute().collect().next();
            Assert.fail("Expected IllegalArgumentException");
        } catch (Exception e) {
            Throwable exception = ExceptionUtils.getRootCause(e);
            assertEquals(IllegalArgumentException.class, exception.getClass());
            assertEquals("Feature vectors should be of equal length.", exception.getMessage());
        }
    }

    @Test
    public void testSaveLoad() throws Exception {
        estimator =
                TestUtils.saveAndReload(
                        tEnv,
                        estimator,
                        tempFolder.newFolder().getAbsolutePath(),
                        NaiveBayes::load);

        NaiveBayesModel model = estimator.fit(trainTable);

        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        NaiveBayesModel::load);

        Table outputTable = model.transform(predictTable)[0];

        Map<IntDoubleVector, Double> actualOutput =
                executeAndCollect(outputTable, model.getFeaturesCol(), model.getPredictionCol());
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testGetModelData() throws Exception {
        List<Row> trainData =
                Arrays.asList(
                        Row.of(Vectors.dense(1, 1.), 11.0), Row.of(Vectors.dense(2, 1.), 11.0));

        trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("features", "label");

        NaiveBayesModel model = estimator.fit(trainTable);

        NaiveBayesModelData actual =
                NaiveBayesModelData.getModelDataStream(model.getModelData()[0])
                        .executeAndCollect()
                        .next();

        assertArrayEquals(new double[] {11.}, actual.labels.toArray(), 1e-5);
        assertArrayEquals(new double[] {0.0}, actual.piArray.toArray(), 1e-5);
        assertEquals(-0.6931471805599453, actual.theta[0][0].get(1.0), 1e-5);
        assertEquals(-0.6931471805599453, actual.theta[0][0].get(2.0), 1e-5);
        assertEquals(0.0, actual.theta[0][1].get(1.0), 1e-5);
    }

    @Test
    public void testSetModelData() {
        NaiveBayesModel modelA = estimator.fit(trainTable);

        Table modelData = modelA.getModelData()[0];
        NaiveBayesModel modelB = new NaiveBayesModel().setModelData(modelData);
        ParamUtils.updateExistingParams(modelB, modelA.getParamMap());

        Table outputTable = modelB.transform(predictTable)[0];

        Map<IntDoubleVector, Double> actualOutput =
                executeAndCollect(outputTable, modelB.getFeaturesCol(), modelB.getPredictionCol());
        assertEquals(expectedOutput, actualOutput);
    }
}
