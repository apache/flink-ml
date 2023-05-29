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

import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelData;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelDataUtil;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelServable;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionWithFtrl;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayInputStream;
import java.io.SequenceInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.apache.flink.ml.util.TestUtils.saveAndLoadServable;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** Tests {@link LogisticRegressionWithFtrl}. */
public class LogisticRegressionWithFtrlTest {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private final double[] expectedCoefficient = new double[] {0.52, -1.21, -1.07, -0.85};
    private static final int MAX_ITER = 100;
    private static final int NUM_SERVERS = 2;
    private static final double TOLERANCE = 1e-7;

    private static final List<Row> trainRows =
            Arrays.asList(
                    Row.of(Tuple2.of(new long[] {0, 1}, new double[] {1, 2}), 0., 1.),
                    Row.of(Tuple2.of(new long[] {0, 2}, new double[] {2, 3}), 0., 2.),
                    Row.of(Tuple2.of(new long[] {0, 3}, new double[] {3, 4}), 0., 3.),
                    Row.of(Tuple2.of(new long[] {0, 2}, new double[] {4, 4}), 0., 4.),
                    Row.of(Tuple2.of(new long[] {0, 1}, new double[] {5, 4}), 0., 5.),
                    Row.of(Tuple2.of(new long[] {0, 2}, new double[] {11, 3}), 1., 1.),
                    Row.of(Tuple2.of(new long[] {0, 3}, new double[] {12, 4}), 1., 2.),
                    Row.of(Tuple2.of(new long[] {0, 1}, new double[] {13, 2}), 1., 3.),
                    Row.of(Tuple2.of(new long[] {0, 3}, new double[] {14, 4}), 1., 4.),
                    Row.of(Tuple2.of(new long[] {0, 2}, new double[] {15, 4}), 1., 5.));

    private static final List<Row> testRows =
            Arrays.asList(
                    Row.of(Vectors.sparse(4, new int[] {0, 1}, new double[] {1, 2}), 0., 1.),
                    Row.of(Vectors.sparse(4, new int[] {0, 2}, new double[] {2, 3}), 0., 2.),
                    Row.of(Vectors.sparse(4, new int[] {0, 3}, new double[] {3, 4}), 0., 3.),
                    Row.of(Vectors.sparse(4, new int[] {0, 2}, new double[] {4, 4}), 0., 4.),
                    Row.of(Vectors.sparse(4, new int[] {0, 1}, new double[] {5, 4}), 0., 5.),
                    Row.of(Vectors.sparse(4, new int[] {0, 2}, new double[] {11, 3}), 1., 1.),
                    Row.of(Vectors.sparse(4, new int[] {0, 3}, new double[] {12, 4}), 1., 2.),
                    Row.of(Vectors.sparse(4, new int[] {0, 1}, new double[] {13, 2}), 1., 3.),
                    Row.of(Vectors.sparse(4, new int[] {0, 3}, new double[] {14, 4}), 1., 4.),
                    Row.of(Vectors.sparse(4, new int[] {0, 2}, new double[] {15, 4}), 1., 5.));

    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table trainTable;
    private Table testTable;
    private DataFrame testDataFrame;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        trainTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            new TupleTypeInfo<>(
                                                    PrimitiveArrayTypeInfo
                                                            .LONG_PRIMITIVE_ARRAY_TYPE_INFO,
                                                    PrimitiveArrayTypeInfo
                                                            .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));
        testTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                testRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            SparseVectorTypeInfo.INSTANCE,
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));

        testDataFrame =
                TestUtils.constructDataFrame(
                        new ArrayList<>(Arrays.asList("features", "label", "weight")),
                        new ArrayList<>(
                                Arrays.asList(
                                        DataTypes.VECTOR(BasicType.DOUBLE),
                                        DataTypes.DOUBLE,
                                        DataTypes.DOUBLE)),
                        testRows);
    }

    @Test
    public void testParam() {
        LogisticRegressionWithFtrl logisticRegressionWithFtrl = new LogisticRegressionWithFtrl();
        assertEquals("features", logisticRegressionWithFtrl.getFeaturesCol());
        assertEquals("label", logisticRegressionWithFtrl.getLabelCol());
        assertNull(logisticRegressionWithFtrl.getWeightCol());
        assertEquals(20, logisticRegressionWithFtrl.getMaxIter());
        assertEquals(1e-6, logisticRegressionWithFtrl.getTol(), TOLERANCE);
        assertEquals(32, logisticRegressionWithFtrl.getGlobalBatchSize());
        assertEquals(0, logisticRegressionWithFtrl.getReg(), TOLERANCE);
        assertEquals(0, logisticRegressionWithFtrl.getElasticNet(), TOLERANCE);
        assertEquals("auto", logisticRegressionWithFtrl.getMultiClass());
        assertEquals("prediction", logisticRegressionWithFtrl.getPredictionCol());
        assertEquals("rawPrediction", logisticRegressionWithFtrl.getRawPredictionCol());

        assertEquals(0.1, logisticRegressionWithFtrl.getAlpha(), TOLERANCE);
        assertEquals(0.1, logisticRegressionWithFtrl.getBeta(), TOLERANCE);
        assertEquals(0L, logisticRegressionWithFtrl.getModelDim());
        assertEquals(1, logisticRegressionWithFtrl.getNumServers());

        logisticRegressionWithFtrl
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setWeightCol("test_weight")
                .setMaxIter(1000)
                .setTol(0.001)
                .setGlobalBatchSize(1000)
                .setReg(0.1)
                .setElasticNet(0.5)
                .setMultiClass("binomial")
                .setPredictionCol("test_predictionCol")
                .setRawPredictionCol("test_rawPredictionCol")
                .setAlpha(0.2)
                .setBeta(0.2)
                .setModelDim(10000000L)
                .setNumServers(4);
        assertEquals("test_features", logisticRegressionWithFtrl.getFeaturesCol());
        assertEquals("test_label", logisticRegressionWithFtrl.getLabelCol());
        assertEquals("test_weight", logisticRegressionWithFtrl.getWeightCol());
        assertEquals(1000, logisticRegressionWithFtrl.getMaxIter());
        assertEquals(0.001, logisticRegressionWithFtrl.getTol(), TOLERANCE);
        assertEquals(1000, logisticRegressionWithFtrl.getGlobalBatchSize());
        assertEquals(0.1, logisticRegressionWithFtrl.getReg(), TOLERANCE);
        assertEquals(0.5, logisticRegressionWithFtrl.getElasticNet(), TOLERANCE);
        assertEquals("binomial", logisticRegressionWithFtrl.getMultiClass());
        assertEquals("test_predictionCol", logisticRegressionWithFtrl.getPredictionCol());
        assertEquals("test_rawPredictionCol", logisticRegressionWithFtrl.getRawPredictionCol());

        assertEquals(0.2, logisticRegressionWithFtrl.getAlpha(), TOLERANCE);
        assertEquals(0.2, logisticRegressionWithFtrl.getBeta(), TOLERANCE);
        assertEquals(10000000L, logisticRegressionWithFtrl.getModelDim());
        assertEquals(4, logisticRegressionWithFtrl.getNumServers());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainTable.as("test_features", "test_label", "test_weight");
        LogisticRegressionWithFtrl logisticRegressionWithFtrl =
                new LogisticRegressionWithFtrl()
                        .setFeaturesCol("test_features")
                        .setLabelCol("test_label")
                        .setWeightCol("test_weight")
                        .setPredictionCol("test_predictionCol")
                        .setRawPredictionCol("test_rawPredictionCol");
        Table output = logisticRegressionWithFtrl.fit(trainTable).transform(tempTable)[0];
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
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        // Fix the parallelism as one for stability tests.
        env.setParallelism(1);
        LogisticRegressionWithFtrl logisticRegressionWithFtrl =
                new LogisticRegressionWithFtrl().setMaxIter(MAX_ITER).setNumServers(NUM_SERVERS);
        LogisticRegressionModel model = logisticRegressionWithFtrl.fit(trainTable);
        List<LogisticRegressionModelData> modelData =
                IteratorUtils.toList(
                        LogisticRegressionModelDataUtil.getModelDataStream(model.getModelData()[0])
                                .executeAndCollect());

        assertEquals(NUM_SERVERS, modelData.size());

        modelData.sort(Comparator.comparingLong(o -> o.startIndex));

        double[] collectedCoefficient = new double[4];
        for (LogisticRegressionModelData modelPiece : modelData) {
            int startIndex = (int) modelPiece.startIndex;
            double[] pieceCoeff = modelPiece.coefficient.values;
            System.arraycopy(pieceCoeff, 0, collectedCoefficient, startIndex, pieceCoeff.length);
        }
        assertArrayEquals(expectedCoefficient, collectedCoefficient, 0.1);
    }

    @Test
    public void testFitAndPredict() throws Exception {
        LogisticRegressionWithFtrl logisticRegressionWithFtrl =
                new LogisticRegressionWithFtrl().setMaxIter(MAX_ITER).setNumServers(NUM_SERVERS);
        Table output = logisticRegressionWithFtrl.fit(trainTable).transform(testTable)[0];
        verifyPredictionResult(
                output,
                logisticRegressionWithFtrl.getFeaturesCol(),
                logisticRegressionWithFtrl.getPredictionCol(),
                logisticRegressionWithFtrl.getRawPredictionCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        LogisticRegressionWithFtrl logisticRegressionWithFtrl =
                new LogisticRegressionWithFtrl().setMaxIter(MAX_ITER).setNumServers(NUM_SERVERS);
        logisticRegressionWithFtrl =
                TestUtils.saveAndReload(
                        tEnv,
                        logisticRegressionWithFtrl,
                        tempFolder.newFolder().getAbsolutePath(),
                        LogisticRegressionWithFtrl::load);
        LogisticRegressionModel model = logisticRegressionWithFtrl.fit(trainTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        LogisticRegressionModel::load);
        assertEquals(
                Arrays.asList("coefficient", "startIndex", "endIndex", "modelVersion"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = model.transform(testTable)[0];
        verifyPredictionResult(
                output,
                logisticRegressionWithFtrl.getFeaturesCol(),
                logisticRegressionWithFtrl.getPredictionCol(),
                logisticRegressionWithFtrl.getRawPredictionCol());
    }

    @Test
    public void testSetModelData() throws Exception {
        LogisticRegressionWithFtrl logisticRegressionWithFtrl =
                new LogisticRegressionWithFtrl().setMaxIter(MAX_ITER).setNumServers(NUM_SERVERS);
        LogisticRegressionModel model = logisticRegressionWithFtrl.fit(trainTable);

        LogisticRegressionModel newModel = new LogisticRegressionModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(testTable)[0];
        verifyPredictionResult(
                output,
                logisticRegressionWithFtrl.getFeaturesCol(),
                logisticRegressionWithFtrl.getPredictionCol(),
                logisticRegressionWithFtrl.getRawPredictionCol());
    }

    @Test
    public void testSaveLoadServableAndPredict() throws Exception {
        LogisticRegressionWithFtrl logisticRegressionWithFtrl =
                new LogisticRegressionWithFtrl().setMaxIter(MAX_ITER).setNumServers(NUM_SERVERS);
        LogisticRegressionModel model = logisticRegressionWithFtrl.fit(trainTable);

        LogisticRegressionModelServable servable =
                saveAndLoadServable(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        LogisticRegressionModel::loadServable);

        DataFrame output = servable.transform(testDataFrame);
        verifyPredictionResult(
                output,
                servable.getFeaturesCol(),
                servable.getPredictionCol(),
                servable.getRawPredictionCol());
    }

    @Test
    public void testSetModelDataToServable() throws Exception {
        LogisticRegressionWithFtrl logisticRegressionWithFtrl =
                new LogisticRegressionWithFtrl().setMaxIter(MAX_ITER).setNumServers(NUM_SERVERS);
        LogisticRegressionModel model = logisticRegressionWithFtrl.fit(trainTable);
        List<byte[]> serializedModelData =
                IteratorUtils.toList(
                        LogisticRegressionModelDataUtil.getModelDataByteStream(
                                        model.getModelData()[0])
                                .executeAndCollect());

        LogisticRegressionModelServable servable = new LogisticRegressionModelServable();
        ParamUtils.updateExistingParams(servable, model.getParamMap());

        List<ByteArrayInputStream> modelStreams =
                serializedModelData.stream()
                        .map(ByteArrayInputStream::new)
                        .collect(Collectors.toList());
        servable.setModelData(new SequenceInputStream(Collections.enumeration(modelStreams)));
        DataFrame output = servable.transform(testDataFrame);
        verifyPredictionResult(
                output,
                servable.getFeaturesCol(),
                servable.getPredictionCol(),
                servable.getRawPredictionCol());
    }

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
}
