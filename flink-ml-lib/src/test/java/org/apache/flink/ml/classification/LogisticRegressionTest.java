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

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.StageTestUtils;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
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
import static org.junit.Assert.fail;

/** Tests {@link LogisticRegression} and {@link LogisticRegressionModel}. */
public class LogisticRegressionTest {

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
            new double[] {0.528, -0.286, -0.429, -0.572};

    private static final double TOLERANCE = 1e-7;

    private Table binomialDataTable;

    private Table multinomialDataTable;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        Collections.shuffle(binomialTrainData);
        binomialDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                binomialTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(DenseVector.class),
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
                                            TypeInformation.of(DenseVector.class),
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));
    }

    @SuppressWarnings("ConstantConditions")
    private void verifyPredictionResult(
            Table output, String featuresCol, String predictionCol, String rawPredictionCol)
            throws Exception {
        List<Row> predResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        for (Row predictionRow : predResult) {
            DenseVector feature = (DenseVector) predictionRow.getField(featuresCol);
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

    @Test
    public void testParam() {
        LogisticRegression logisticRegression = new LogisticRegression();
        assertEquals(logisticRegression.getLabelCol(), "label");
        assertNull(logisticRegression.getWeightCol());
        assertEquals(logisticRegression.getMaxIter(), 20);
        assertEquals(logisticRegression.getReg(), 0, TOLERANCE);
        assertEquals(logisticRegression.getLearningRate(), 0.1, TOLERANCE);
        assertEquals(logisticRegression.getGlobalBatchSize(), 32);
        assertEquals(logisticRegression.getTol(), 1e-6, TOLERANCE);
        assertEquals(logisticRegression.getMultiClass(), "auto");
        assertEquals(logisticRegression.getFeaturesCol(), "features");
        assertEquals(logisticRegression.getPredictionCol(), "prediction");
        assertEquals(logisticRegression.getRawPredictionCol(), "rawPrediction");

        logisticRegression
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setWeightCol("test_weight")
                .setMaxIter(1000)
                .setTol(0.001)
                .setLearningRate(0.5)
                .setGlobalBatchSize(1000)
                .setReg(0.1)
                .setMultiClass("binomial")
                .setPredictionCol("test_predictionCol")
                .setRawPredictionCol("test_rawPredictionCol");
        assertEquals(logisticRegression.getFeaturesCol(), "test_features");
        assertEquals(logisticRegression.getLabelCol(), "test_label");
        assertEquals(logisticRegression.getWeightCol(), "test_weight");
        assertEquals(logisticRegression.getMaxIter(), 1000);
        assertEquals(logisticRegression.getTol(), 0.001, TOLERANCE);
        assertEquals(logisticRegression.getLearningRate(), 0.5, TOLERANCE);
        assertEquals(logisticRegression.getGlobalBatchSize(), 1000);
        assertEquals(logisticRegression.getReg(), 0.1, TOLERANCE);
        assertEquals(logisticRegression.getMultiClass(), "binomial");
        assertEquals(logisticRegression.getPredictionCol(), "test_predictionCol");
        assertEquals(logisticRegression.getRawPredictionCol(), "test_rawPredictionCol");
    }

    @Test
    public void testFeaturePredictionParam() {
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
    public void testSaveLoadAndPredict() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        logisticRegression =
                StageTestUtils.saveAndReload(
                        env, logisticRegression, tempFolder.newFolder().getAbsolutePath());
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);
        model = StageTestUtils.saveAndReload(env, model, tempFolder.newFolder().getAbsolutePath());
        assertEquals(
                Arrays.asList("coefficient", "versionId", "isLastRecord"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = model.transform(binomialDataTable)[0];
        verifyPredictionResult(
                output,
                logisticRegression.getFeaturesCol(),
                logisticRegression.getPredictionCol(),
                logisticRegression.getRawPredictionCol());
    }

    @Test
    public void testGetModelData() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);
        LogisticRegressionModelData modelData =
                LogisticRegressionModelData.getModelDataStream(model.getModelData()[0])
                        .executeAndCollect()
                        .next();
        assertNotNull(modelData);
        assertArrayEquals(expectedCoefficient, modelData.coefficient.values, 0.1);
    }

    @Test
    public void testSetModelData() throws Exception {
        LogisticRegression logisticRegression = new LogisticRegression().setWeightCol("weight");
        LogisticRegressionModel model = logisticRegression.fit(binomialDataTable);

        LogisticRegressionModel newModel = new LogisticRegressionModel();
        ReadWriteUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(binomialDataTable)[0];
        verifyPredictionResult(
                output,
                logisticRegression.getFeaturesCol(),
                logisticRegression.getPredictionCol(),
                logisticRegression.getRawPredictionCol());
    }

    @Test
    public void testMultinomialFit() {
        try {
            new LogisticRegression().fit(multinomialDataTable);
            env.execute();
            fail();
        } catch (Exception e) {
            assertEquals(
                    "Multinomial classification is not supported yet. Supported options: [auto, binomial].",
                    e.getCause().getCause().getMessage());
        }
    }
}
