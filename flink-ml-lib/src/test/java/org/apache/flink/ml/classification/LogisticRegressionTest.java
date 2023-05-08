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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelData;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelDataUtil;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelServable;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.RandomUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.flink.ml.util.TestUtils.saveAndLoadServable;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Tests {@link LogisticRegression} and {@link LogisticRegressionModel}. */
public class LogisticRegressionTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private static final List<Row> binomialTrainData =
            Arrays.asList(
                    Row.of(Vectors.dense(1, 2, 3, 4), 0., 1.),
                    Row.of(Vectors.dense(2, 2, 3, 4), 0., 2.),
                    Row.of(Vectors.dense(3, 2, 3, 4), 0., 3.),
                    Row.of(Vectors.dense(4, 2, 3, 4), 0., 4.),
                    Row.of(Vectors.dense(5, 2, 3, 4), 0., 5.),
                    Row.of(Vectors.dense(11, 2, 3, 4), 1., 1.),
                    Row.of(Vectors.dense(12, 2, 3, 4), 1., 2.),
                    Row.of(Vectors.dense(13, 2, 3, 4), 1., 3.),
                    Row.of(Vectors.dense(14, 2, 3, 4), 1., 4.),
                    Row.of(Vectors.dense(15, 2, 3, 4), 1., 5.));

    private static final List<Row> multinomialTrainData =
            Arrays.asList(
                    Row.of(Vectors.dense(1, 2, 3, 4), 0., 1.),
                    Row.of(Vectors.dense(2, 2, 3, 4), 0., 2.),
                    Row.of(Vectors.dense(3, 2, 3, 4), 2., 3.),
                    Row.of(Vectors.dense(4, 2, 3, 4), 2., 4.),
                    Row.of(Vectors.dense(5, 2, 3, 4), 2., 5.),
                    Row.of(Vectors.dense(11, 2, 3, 4), 1., 1.),
                    Row.of(Vectors.dense(12, 2, 3, 4), 1., 2.),
                    Row.of(Vectors.dense(13, 2, 3, 4), 1., 3.),
                    Row.of(Vectors.dense(14, 2, 3, 4), 1., 4.),
                    Row.of(Vectors.dense(15, 2, 3, 4), 1., 5.));

    private static final double[] expectedCoefficient =
            new double[] {0.525, -0.283, -0.425, -0.567};

    private static final double TOLERANCE = 1e-7;

    private Table binomialDataTable;
    private Table binomialSparseDataTable;

    private Table multinomialDataTable;

    private DataFrame binomialDataDataFrame;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        Collections.shuffle(binomialTrainData);
        binomialDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                binomialTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            DenseVectorTypeInfo.INSTANCE, Types.DOUBLE, Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));

        List<Row> binomialSparseTrainData =
                binomialTrainData.stream()
                        .map(
                                r -> {
                                    DenseVector features = r.getFieldAs(0);
                                    double label = r.getFieldAs(1);
                                    double weight = r.getFieldAs(2);
                                    return Row.of(features.toSparse(), label, weight);
                                })
                        .collect(Collectors.toList());
        binomialSparseDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                binomialSparseTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            SparseVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));

        multinomialDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                multinomialTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            DenseVectorTypeInfo.INSTANCE, Types.DOUBLE, Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));
        binomialDataDataFrame =
                TestUtils.constructDataFrame(
                        new ArrayList<>(Arrays.asList("features", "label", "weight")),
                        new ArrayList<>(
                                Arrays.asList(
                                        DataTypes.VECTOR(BasicType.DOUBLE),
                                        DataTypes.DOUBLE,
                                        DataTypes.DOUBLE)),
                        binomialTrainData);
    }

    @SuppressWarnings("ConstantConditions, unchecked")
    private void verifyPredictionResult(
            Table output, String featuresCol, String predictionCol, String rawPredictionCol)
            throws Exception {
        List<Row> predResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        for (Row predictionRow : predResult) {
            DenseVector feature = ((Vector) predictionRow.getField(featuresCol)).toDense();
            double prediction = (double) predictionRow.getField(predictionCol);
            DenseVector rawPrediction = (DenseVector) predictionRow.getField(rawPredictionCol);
            if (feature.get(0) <= 5) {
                assertEquals(0, prediction, TOLERANCE);
                assertTrue(rawPrediction.get(0) > 0.5);
            } else {
                assertEquals(1, prediction, TOLERANCE);
                assertTrue(rawPrediction.get(0) < 0.5);
            }
        }
    }

    private void verifyPredictionResult(
            DataFrame output, String featuresCol, String predictionCol, String rawPredictionCol) {
        int featuresColIndex = output.getIndex(featuresCol);
        int predictionColIndex = output.getIndex(predictionCol);
        int rawPredictionColIndex = output.getIndex(rawPredictionCol);

        for (org.apache.flink.ml.servable.api.Row predictionRow : output.collect()) {
            DenseVector feature = ((Vector) predictionRow.get(featuresColIndex)).toDense();
            double prediction = (double) predictionRow.get(predictionColIndex);
            DenseVector rawPrediction = (DenseVector) predictionRow.get(rawPredictionColIndex);
            if (feature.get(0) <= 5) {
                assertEquals(0, prediction, TOLERANCE);
                assertTrue(rawPrediction.get(0) > 0.5);
            } else {
                assertEquals(1, prediction, TOLERANCE);
                assertTrue(rawPrediction.get(0) < 0.5);
            }
        }
    }

    @Test
    public void testParam() {
        LogisticRegression logisticRegression = new LogisticRegression();
        assertEquals("features", logisticRegression.getFeaturesCol());
        assertEquals("label", logisticRegression.getLabelCol());
        assertNull(logisticRegression.getWeightCol());
        assertEquals(20, logisticRegression.getMaxIter());
        assertEquals(1e-6, logisticRegression.getTol(), TOLERANCE);
        assertEquals(0.1, logisticRegression.getLearningRate(), TOLERANCE);
        assertEquals(32, logisticRegression.getGlobalBatchSize());
        assertEquals(0, logisticRegression.getReg(), TOLERANCE);
        assertEquals(0, logisticRegression.getElasticNet(), TOLERANCE);
        assertEquals("auto", logisticRegression.getMultiClass());
        assertEquals("prediction", logisticRegression.getPredictionCol());
        assertEquals("rawPrediction", logisticRegression.getRawPredictionCol());

        logisticRegression
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setWeightCol("test_weight")
                .setMaxIter(1000)
                .setTol(0.001)
                .setLearningRate(0.5)
                .setGlobalBatchSize(1000)
                .setReg(0.1)
                .setElasticNet(0.5)
                .setMultiClass("binomial")
                .setPredictionCol("test_predictionCol")
                .setRawPredictionCol("test_rawPredictionCol");
        assertEquals("test_features", logisticRegression.getFeaturesCol());
        assertEquals("test_label", logisticRegression.getLabelCol());
        assertEquals("test_weight", logisticRegression.getWeightCol());
        assertEquals(1000, logisticRegression.getMaxIter());
        assertEquals(0.001, logisticRegression.getTol(), TOLERANCE);
        assertEquals(0.5, logisticRegression.getLearningRate(), TOLERANCE);
        assertEquals(1000, logisticRegression.getGlobalBatchSize());
        assertEquals(0.1, logisticRegression.getReg(), TOLERANCE);
        assertEquals(0.5, logisticRegression.getElasticNet(), TOLERANCE);
        assertEquals("binomial", logisticRegression.getMultiClass());
        assertEquals("test_predictionCol", logisticRegression.getPredictionCol());
        assertEquals("test_rawPredictionCol", logisticRegression.getRawPredictionCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = binomialDataTable.as("test_features", "test_label", "test_weight");
        LogisticRegression logisticRegression =
                new LogisticRegression()
                        .setFeaturesCol("test_features")
                        .setLabelCol("test_label")
                        .setWeightCol("test_weight")
                        .setPredictionCol("test_predictionCol")
                        .setRawPredictionCol("test_rawPredictionCol");
        Table output = logisticRegression.fit(binomialDataTable).transform(tempTable)[0];
        assertEquals(
                Arrays.asList(
                        "test_features",
                        "test_label",
                        "test_weight",
                        "test_predictionCol",
                        "test_rawPredictionCol"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        Table output = logisticRegression.fit(binomialDataTable).transform(binomialDataTable)[0];
        verifyPredictionResult(
                output,
                logisticRegression.getFeaturesCol(),
                logisticRegression.getPredictionCol(),
                logisticRegression.getRawPredictionCol());
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        binomialDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, binomialDataTable);
        assertArrayEquals(
                new Class<?>[] {SparseVector.class, Integer.class, Integer.class},
                TestUtils.getColumnDataTypes(binomialDataTable));

        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        Table output = logisticRegression.fit(binomialDataTable).transform(binomialDataTable)[0];
        verifyPredictionResult(
                output,
                logisticRegression.getFeaturesCol(),
                logisticRegression.getPredictionCol(),
                logisticRegression.getRawPredictionCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        logisticRegression =
                TestUtils.saveAndReload(
                        tEnv,
                        logisticRegression,
                        tempFolder.newFolder().getAbsolutePath(),
                        LogisticRegression::load);
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        LogisticRegressionModel::load);
        assertEquals(
                Arrays.asList("coefficient", "modelVersion"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = model.transform(binomialDataTable)[0];
        verifyPredictionResult(
                output,
                logisticRegression.getFeaturesCol(),
                logisticRegression.getPredictionCol(),
                logisticRegression.getRawPredictionCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);
        List<LogisticRegressionModelData> modelData =
                IteratorUtils.toList(
                        LogisticRegressionModelDataUtil.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());
        assertEquals(1, modelData.size());
        assertArrayEquals(expectedCoefficient, modelData.get(0).coefficient.values, 0.1);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelDataFromSparseInput() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        LogisticRegressionModel model = logisticRegression.fit(binomialSparseDataTable);
        List<LogisticRegressionModelData> modelData =
                IteratorUtils.toList(
                        LogisticRegressionModelDataUtil.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());
        assertEquals(1, modelData.size());
        assertArrayEquals(expectedCoefficient, modelData.get(0).coefficient.values, 0.1);
    }

    @Test
    public void testSetModelData() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);

        LogisticRegressionModel newModel = new LogisticRegressionModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(binomialDataTable)[0];
        verifyPredictionResult(
                output,
                logisticRegression.getFeaturesCol(),
                logisticRegression.getPredictionCol(),
                logisticRegression.getRawPredictionCol());
    }

    @Test
    public void testSaveLoadServableAndPredict() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);

        LogisticRegressionModelServable servable =
                saveAndLoadServable(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        LogisticRegressionModel::loadServable);

        DataFrame output = servable.transform(binomialDataDataFrame);
        verifyPredictionResult(
                output,
                servable.getFeaturesCol(),
                servable.getPredictionCol(),
                servable.getRawPredictionCol());
    }

    @Test
    public void testSetModelDataToServable() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);
        byte[] serializedModelData =
                LogisticRegressionModelDataUtil.getModelDataByteStream(model.getModelData()[0])
                        .executeAndCollect()
                        .next();

        LogisticRegressionModelServable servable = new LogisticRegressionModelServable();
        ParamUtils.updateExistingParams(servable, model.getParamMap());
        servable.setModelData(new ByteArrayInputStream(serializedModelData));

        DataFrame output = servable.transform(binomialDataDataFrame);
        verifyPredictionResult(
                output,
                servable.getFeaturesCol(),
                servable.getPredictionCol(),
                servable.getRawPredictionCol());
    }

    @Test
    public void testMultinomialFit() {
        try {
            new LogisticRegression().fit(multinomialDataTable);
            env.execute();
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "Multinomial classification is not supported yet. Supported options: [auto, binomial].",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testMoreSubtaskThanData() throws Exception {
        List<Row> binomialTrainData =
                Arrays.asList(
                        Row.of(Vectors.dense(1, 2, 3, 4), 0., 1.),
                        Row.of(Vectors.dense(11, 2, 3, 4), 1., 1.));

        Table binomialDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                binomialTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            DenseVectorTypeInfo.INSTANCE, Types.DOUBLE, Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));

        LogisticRegression logisticRegression =
                new LogisticRegression().setWeightCol("weight").setGlobalBatchSize(128);
        Table output = logisticRegression.fit(binomialDataTable).transform(binomialDataTable)[0];
        verifyPredictionResult(
                output,
                logisticRegression.getFeaturesCol(),
                logisticRegression.getPredictionCol(),
                logisticRegression.getRawPredictionCol());
    }

    @Test
    public void testRegularization() throws Exception {
        checkRegularization(0, RandomUtils.nextDouble(0, 1), expectedCoefficient);
        checkRegularization(0.1, 0, new double[] {0.484, -0.258, -0.388, -0.517});
        checkRegularization(0.1, 1, new double[] {0.417, -0.145, -0.312, -0.480});
        checkRegularization(0.1, 0.5, new double[] {0.451, -0.203, -0.351, -0.498});
    }

    @SuppressWarnings("unchecked")
    private void checkRegularization(double reg, double elasticNet, double[] expectedCoefficient)
            throws Exception {
        LogisticRegressionModel model =
                new LogisticRegression()
                        .setWeightCol("weight")
                        .setReg(reg)
                        .setElasticNet(elasticNet)
                        .fit(binomialDataTable);
        List<LogisticRegressionModelData> modelData =
                IteratorUtils.toList(
                        LogisticRegressionModelDataUtil.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());
        final double errorTol = 1e-3;
        assertArrayEquals(expectedCoefficient, modelData.get(0).coefficient.values, errorTol);
    }
}
