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
import org.apache.flink.ml.classification.linearsvc.LinearSVC;
import org.apache.flink.ml.classification.linearsvc.LinearSVCModel;
import org.apache.flink.ml.classification.linearsvc.LinearSVCModelData;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
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
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** Tests {@link LinearSVC} and {@link LinearSVCModel}. */
public class LinearSVCTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private static final List<Row> trainData =
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

    private static final double[] expectedCoefficient =
            new double[] {0.470, -0.273, -0.410, -0.546};

    private static final double TOLERANCE = 1e-7;

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

    @SuppressWarnings("ConstantConditions, unchecked")
    private void verifyPredictionResult(
            Table output, String featuresCol, String predictionCol, String rawPredictionCol)
            throws Exception {
        List<Row> predResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        for (Row predictionRow : predResult) {
            DenseIntDoubleVector feature =
                    ((IntDoubleVector) predictionRow.getField(featuresCol)).toDense();
            double prediction = (Double) predictionRow.getField(predictionCol);
            DenseIntDoubleVector rawPrediction =
                    (DenseIntDoubleVector) predictionRow.getField(rawPredictionCol);
            if (feature.get(0) <= 5) {
                assertEquals(0, prediction, TOLERANCE);
                assertTrue(rawPrediction.get(0) < 0);
            } else {
                assertEquals(1, prediction, TOLERANCE);
                assertTrue(rawPrediction.get(0) > 0);
            }
        }
    }

    @Test
    public void testParam() {
        LinearSVC linearSVC = new LinearSVC();
        assertEquals("features", linearSVC.getFeaturesCol());
        assertEquals("label", linearSVC.getLabelCol());
        assertNull(linearSVC.getWeightCol());
        assertEquals(20, linearSVC.getMaxIter());
        assertEquals(1e-6, linearSVC.getTol(), TOLERANCE);
        assertEquals(0.1, linearSVC.getLearningRate(), TOLERANCE);
        assertEquals(32, linearSVC.getGlobalBatchSize());
        assertEquals(0, linearSVC.getReg(), TOLERANCE);
        assertEquals(0, linearSVC.getElasticNet(), TOLERANCE);
        assertEquals(0.0, linearSVC.getThreshold(), TOLERANCE);
        assertEquals("prediction", linearSVC.getPredictionCol());
        assertEquals("rawPrediction", linearSVC.getRawPredictionCol());

        linearSVC
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setWeightCol("test_weight")
                .setMaxIter(1000)
                .setTol(0.001)
                .setLearningRate(0.5)
                .setGlobalBatchSize(1000)
                .setReg(0.1)
                .setElasticNet(0.5)
                .setThreshold(0.5)
                .setPredictionCol("test_predictionCol")
                .setRawPredictionCol("test_rawPredictionCol");
        assertEquals("test_features", linearSVC.getFeaturesCol());
        assertEquals("test_label", linearSVC.getLabelCol());
        assertEquals("test_weight", linearSVC.getWeightCol());
        assertEquals(1000, linearSVC.getMaxIter());
        assertEquals(0.001, linearSVC.getTol(), TOLERANCE);
        assertEquals(0.5, linearSVC.getLearningRate(), TOLERANCE);
        assertEquals(1000, linearSVC.getGlobalBatchSize());
        assertEquals(0.1, linearSVC.getReg(), TOLERANCE);
        assertEquals(0.5, linearSVC.getElasticNet(), TOLERANCE);
        assertEquals(0.5, linearSVC.getThreshold(), TOLERANCE);
        assertEquals("test_predictionCol", linearSVC.getPredictionCol());
        assertEquals("test_rawPredictionCol", linearSVC.getRawPredictionCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainDataTable.as("test_features", "test_label", "test_weight");
        LinearSVC linearSVC =
                new LinearSVC()
                        .setFeaturesCol("test_features")
                        .setLabelCol("test_label")
                        .setWeightCol("test_weight")
                        .setPredictionCol("test_predictionCol")
                        .setRawPredictionCol("test_rawPredictionCol");
        Table output = linearSVC.fit(trainDataTable).transform(tempTable)[0];
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
        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight");
        Table output = linearSVC.fit(trainDataTable).transform(trainDataTable)[0];
        verifyPredictionResult(
                output,
                linearSVC.getFeaturesCol(),
                linearSVC.getPredictionCol(),
                linearSVC.getRawPredictionCol());
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        trainDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, trainDataTable);
        assertArrayEquals(
                new Class<?>[] {SparseIntDoubleVector.class, Integer.class, Integer.class},
                TestUtils.getColumnDataTypes(trainDataTable));

        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight");
        Table output = linearSVC.fit(trainDataTable).transform(trainDataTable)[0];
        verifyPredictionResult(
                output,
                linearSVC.getFeaturesCol(),
                linearSVC.getPredictionCol(),
                linearSVC.getRawPredictionCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight");
        linearSVC =
                TestUtils.saveAndReload(
                        tEnv, linearSVC, tempFolder.newFolder().getAbsolutePath(), LinearSVC::load);
        LinearSVCModel model = linearSVC.fit(trainDataTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        LinearSVCModel::load);
        assertEquals(
                Collections.singletonList("coefficient"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = model.transform(trainDataTable)[0];
        verifyPredictionResult(
                output,
                linearSVC.getFeaturesCol(),
                linearSVC.getPredictionCol(),
                linearSVC.getRawPredictionCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight");
        LinearSVCModel model = linearSVC.fit(trainDataTable);
        List<LinearSVCModelData> modelData =
                IteratorUtils.toList(
                        LinearSVCModelData.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());
        assertEquals(1, modelData.size());
        assertArrayEquals(expectedCoefficient, modelData.get(0).coefficient.values, 0.1);
    }

    @Test
    public void testSetModelData() throws Exception {
        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight");
        LinearSVCModel model = linearSVC.fit(trainDataTable);

        LinearSVCModel newModel = new LinearSVCModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(trainDataTable)[0];
        verifyPredictionResult(
                output,
                linearSVC.getFeaturesCol(),
                linearSVC.getPredictionCol(),
                linearSVC.getRawPredictionCol());
    }

    @Test
    public void testMoreSubtaskThanData() throws Exception {
        List<Row> trainData =
                Arrays.asList(
                        Row.of(Vectors.dense(1, 2, 3, 4), 0., 1.),
                        Row.of(Vectors.dense(11, 2, 3, 4), 1., 1.));

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

        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight").setGlobalBatchSize(128);
        Table output = linearSVC.fit(trainDataTable).transform(trainDataTable)[0];
        verifyPredictionResult(
                output,
                linearSVC.getFeaturesCol(),
                linearSVC.getPredictionCol(),
                linearSVC.getRawPredictionCol());
    }

    @Test
    public void testRegularization() throws Exception {
        checkRegularization(0, RandomUtils.nextDouble(0, 1), expectedCoefficient);
        checkRegularization(0.1, 0, new double[] {0.437, -0.262, -0.393, -0.524});
        checkRegularization(0.1, 1, new double[] {0.426, -0.197, -0.329, -0.463});
        checkRegularization(0.1, 0.5, new double[] {0.419, -0.238, -0.372, -0.505});
    }

    @Test
    public void testThreshold() throws Exception {
        checkThreshold(-Double.MAX_VALUE, 1);
        checkThreshold(Double.MAX_VALUE, 0);
    }

    @SuppressWarnings("unchecked")
    private void checkRegularization(double reg, double elasticNet, double[] expectedCoefficient)
            throws Exception {
        LinearSVCModel model =
                new LinearSVC()
                        .setWeightCol("weight")
                        .setReg(reg)
                        .setElasticNet(elasticNet)
                        .fit(trainDataTable);
        List<LinearSVCModelData> modelData =
                IteratorUtils.toList(
                        LinearSVCModelData.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());
        final double errorTol = 1e-3;
        assertArrayEquals(expectedCoefficient, modelData.get(0).coefficient.values, errorTol);
    }

    @SuppressWarnings("unchecked")
    private void checkThreshold(double threshold, double target) throws Exception {
        LinearSVC linearSVC = new LinearSVC().setWeightCol("weight");

        Table predictions =
                linearSVC.setThreshold(threshold).fit(trainDataTable).transform(trainDataTable)[0];

        List<Row> predResult =
                IteratorUtils.toList(tEnv.toDataStream(predictions).executeAndCollect());
        for (Row r : predResult) {
            assertEquals(target, r.getField(linearSVC.getPredictionCol()));
        }
    }
}
