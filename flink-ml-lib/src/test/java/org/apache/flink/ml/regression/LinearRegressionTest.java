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

package org.apache.flink.ml.regression;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.regression.linearregression.LinearRegression;
import org.apache.flink.ml.regression.linearregression.LinearRegressionModel;
import org.apache.flink.ml.regression.linearregression.LinearRegressionModelData;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.RandomUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** Tests {@link LinearRegression} and {@link LinearRegressionModel}. */
public class LinearRegressionTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private static final List<Row> trainData =
            Arrays.asList(
                    Row.of(Vectors.dense(2, 1), 4.0, 1.0),
                    Row.of(Vectors.dense(3, 2), 7.0, 1.0),
                    Row.of(Vectors.dense(4, 3), 10.0, 1.0),
                    Row.of(Vectors.dense(2, 4), 10.0, 1.0),
                    Row.of(Vectors.dense(2, 2), 6.0, 1.0),
                    Row.of(Vectors.dense(4, 3), 10.0, 1.0),
                    Row.of(Vectors.dense(1, 2), 5.0, 1.0),
                    Row.of(Vectors.dense(5, 3), 11.0, 1.0));

    private static final double[] expectedCoefficient = new double[] {1.141, 1.829};

    private static final double TOLERANCE = 1e-7;

    private static final double PREDICTION_TOLERANCE = 0.1;

    private static final double COEFFICIENT_TOLERANCE = 0.1;

    private Table trainDataTable;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        Collections.shuffle(trainData);
        trainDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            DenseIntDoubleVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(Table output, String labelCol, String predictionCol)
            throws Exception {
        List<Row> predResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        for (Row predictionRow : predResult) {
            double label = ((Number) predictionRow.getField(labelCol)).doubleValue();
            double prediction = (double) predictionRow.getField(predictionCol);
            assertTrue(Math.abs(prediction - label) / label < PREDICTION_TOLERANCE);
        }
    }

    @Test
    public void testParam() {
        LinearRegression linearRegression = new LinearRegression();
        assertEquals("features", linearRegression.getFeaturesCol());
        assertEquals("label", linearRegression.getLabelCol());
        assertNull(linearRegression.getWeightCol());
        assertEquals(20, linearRegression.getMaxIter());
        assertEquals(1e-6, linearRegression.getTol(), TOLERANCE);
        assertEquals(0.1, linearRegression.getLearningRate(), TOLERANCE);
        assertEquals(32, linearRegression.getGlobalBatchSize());
        assertEquals(0, linearRegression.getReg(), TOLERANCE);
        assertEquals(0, linearRegression.getElasticNet(), TOLERANCE);
        assertEquals("prediction", linearRegression.getPredictionCol());

        linearRegression
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setWeightCol("test_weight")
                .setMaxIter(1000)
                .setTol(0.001)
                .setLearningRate(0.5)
                .setGlobalBatchSize(1000)
                .setReg(0.1)
                .setElasticNet(0.5)
                .setPredictionCol("test_predictionCol");
        assertEquals("test_features", linearRegression.getFeaturesCol());
        assertEquals("test_label", linearRegression.getLabelCol());
        assertEquals("test_weight", linearRegression.getWeightCol());
        assertEquals(1000, linearRegression.getMaxIter());
        assertEquals(0.001, linearRegression.getTol(), TOLERANCE);
        assertEquals(0.5, linearRegression.getLearningRate(), TOLERANCE);
        assertEquals(1000, linearRegression.getGlobalBatchSize());
        assertEquals(0.1, linearRegression.getReg(), TOLERANCE);
        assertEquals(0.5, linearRegression.getElasticNet(), TOLERANCE);
        assertEquals("test_predictionCol", linearRegression.getPredictionCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainDataTable.as("test_features", "test_label", "test_weight");
        LinearRegression linearRegression =
                new LinearRegression()
                        .setFeaturesCol("test_features")
                        .setLabelCol("test_label")
                        .setWeightCol("test_weight")
                        .setPredictionCol("test_predictionCol");
        Table output = linearRegression.fit(trainDataTable).transform(tempTable)[0];
        assertEquals(
                Arrays.asList("test_features", "test_label", "test_weight", "test_predictionCol"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        LinearRegression linearRegression = new LinearRegression().setWeightCol("weight");
        Table output = linearRegression.fit(trainDataTable).transform(trainDataTable)[0];
        verifyPredictionResult(
                output, linearRegression.getLabelCol(), linearRegression.getPredictionCol());
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        trainDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, trainDataTable);
        assertArrayEquals(
                new Class<?>[] {SparseIntDoubleVector.class, Integer.class, Integer.class},
                TestUtils.getColumnDataTypes(trainDataTable));

        LinearRegression linearRegression = new LinearRegression().setWeightCol("weight");
        Table output = linearRegression.fit(trainDataTable).transform(trainDataTable)[0];
        verifyPredictionResult(
                output, linearRegression.getLabelCol(), linearRegression.getPredictionCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        LinearRegression linearRegression = new LinearRegression().setWeightCol("weight");
        linearRegression =
                TestUtils.saveAndReload(
                        tEnv,
                        linearRegression,
                        tempFolder.newFolder().getAbsolutePath(),
                        LinearRegression::load);
        LinearRegressionModel model = linearRegression.fit(trainDataTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        LinearRegressionModel::load);
        assertEquals(
                Collections.singletonList("coefficient"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = model.transform(trainDataTable)[0];
        verifyPredictionResult(
                output, linearRegression.getLabelCol(), linearRegression.getPredictionCol());
    }

    @Test
    public void testGetModelData() throws Exception {
        LinearRegression linearRegression = new LinearRegression().setWeightCol("weight");
        LinearRegressionModel model = linearRegression.fit(trainDataTable);
        List<LinearRegressionModelData> modelData =
                IteratorUtils.toList(
                        LinearRegressionModelData.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());
        assertNotNull(modelData);
        assertEquals(1, modelData.size());
        assertArrayEquals(
                expectedCoefficient, modelData.get(0).coefficient.values, COEFFICIENT_TOLERANCE);
    }

    @Test
    public void testSetModelData() throws Exception {
        LinearRegression linearRegression = new LinearRegression().setWeightCol("weight");
        LinearRegressionModel model = linearRegression.fit(trainDataTable);

        LinearRegressionModel newModel = new LinearRegressionModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(trainDataTable)[0];
        verifyPredictionResult(
                output, linearRegression.getLabelCol(), linearRegression.getPredictionCol());
    }

    @Test
    public void testMoreSubtaskThanData() throws Exception {
        List<Row> trainData =
                Arrays.asList(
                        Row.of(Vectors.dense(2, 1), 4.0, 1.0),
                        Row.of(Vectors.dense(3, 2), 7.0, 1.0));

        Table trainDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            DenseIntDoubleVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));

        LinearRegression linearRegression =
                new LinearRegression().setWeightCol("weight").setGlobalBatchSize(128);
        Table output = linearRegression.fit(trainDataTable).transform(trainDataTable)[0];
        verifyPredictionResult(
                output, linearRegression.getLabelCol(), linearRegression.getPredictionCol());
    }

    @Test
    public void testRegularization() throws Exception {
        checkRegularization(0, RandomUtils.nextDouble(0, 1), expectedCoefficient);
        checkRegularization(0.1, 0, new double[] {1.165, 1.780});
        checkRegularization(0.1, 1, new double[] {1.143, 1.812});
        checkRegularization(0.1, 0.5, new double[] {1.154, 1.796});
    }

    @SuppressWarnings("unchecked")
    private void checkRegularization(double reg, double elasticNet, double[] expectedCoefficient)
            throws Exception {
        LinearRegressionModel model =
                new LinearRegression()
                        .setWeightCol("weight")
                        .setReg(reg)
                        .setElasticNet(elasticNet)
                        .fit(trainDataTable);
        List<LinearRegressionModelData> modelData =
                IteratorUtils.toList(
                        LinearRegressionModelData.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());
        final double errorTol = 1e-3;
        assertArrayEquals(expectedCoefficient, modelData.get(0).coefficient.values, errorTol);
    }
}
