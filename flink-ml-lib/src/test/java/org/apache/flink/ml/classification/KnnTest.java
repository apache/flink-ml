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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.classification.knn.Knn;
import org.apache.flink.ml.classification.knn.KnnModel;
import org.apache.flink.ml.classification.knn.KnnModelData;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link Knn} and {@link KnnModel}. */
public class KnnTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainData;
    private Table predictData;
    private static final List<Row> trainRows =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(2.0, 3.0), 1.0),
                            Row.of(Vectors.dense(2.1, 3.1), 1.0),
                            Row.of(Vectors.dense(200.1, 300.1), 2.0),
                            Row.of(Vectors.dense(200.2, 300.2), 2.0),
                            Row.of(Vectors.dense(200.3, 300.3), 2.0),
                            Row.of(Vectors.dense(200.4, 300.4), 2.0),
                            Row.of(Vectors.dense(200.4, 300.4), 2.0),
                            Row.of(Vectors.dense(200.6, 300.6), 2.0),
                            Row.of(Vectors.dense(2.1, 3.1), 1.0),
                            Row.of(Vectors.dense(2.1, 3.1), 1.0),
                            Row.of(Vectors.dense(2.1, 3.1), 1.0),
                            Row.of(Vectors.dense(2.1, 3.1), 1.0),
                            Row.of(Vectors.dense(2.3, 3.2), 1.0),
                            Row.of(Vectors.dense(2.3, 3.2), 1.0),
                            Row.of(Vectors.dense(2.8, 3.2), 3.0),
                            Row.of(Vectors.dense(300., 3.2), 4.0),
                            Row.of(Vectors.dense(2.2, 3.2), 1.0),
                            Row.of(Vectors.dense(2.4, 3.2), 5.0),
                            Row.of(Vectors.dense(2.5, 3.2), 5.0),
                            Row.of(Vectors.dense(2.5, 3.2), 5.0),
                            Row.of(Vectors.dense(2.1, 3.1), 1.0)));
    private static final List<Row> predictRows =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(4.0, 4.1), 5.0),
                            Row.of(Vectors.dense(300, 42), 2.0)));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        Schema schema =
                Schema.newBuilder()
                        .column("f0", DataTypes.of(DenseVector.class))
                        .column("f1", DataTypes.DOUBLE())
                        .build();
        DataStream<Row> dataStream = env.fromCollection(trainRows);
        trainData = tEnv.fromDataStream(dataStream, schema).as("features", "label");
        DataStream<Row> predDataStream = env.fromCollection(predictRows);
        predictData = tEnv.fromDataStream(predDataStream, schema).as("features", "label");
    }

    private static void verifyPredictionResult(Table output, String labelCol, String predictionCol)
            throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Tuple2<Double, Double>> stream =
                tEnv.toDataStream(output)
                        .map(
                                new MapFunction<Row, Tuple2<Double, Double>>() {
                                    @Override
                                    public Tuple2<Double, Double> map(Row row) {
                                        return Tuple2.of(
                                                ((Number) row.getField(labelCol)).doubleValue(),
                                                (Double) row.getField(predictionCol));
                                    }
                                });
        List<Tuple2<Double, Double>> result = IteratorUtils.toList(stream.executeAndCollect());
        for (Tuple2<Double, Double> t2 : result) {
            Assert.assertEquals(t2.f0, t2.f1);
        }
    }

    @Test
    public void testParam() {
        Knn knn = new Knn();
        assertEquals("features", knn.getFeaturesCol());
        assertEquals("label", knn.getLabelCol());
        assertEquals(5, (int) knn.getK());
        assertEquals("prediction", knn.getPredictionCol());
        knn.setLabelCol("test_label")
                .setFeaturesCol("test_features")
                .setK(4)
                .setPredictionCol("test_prediction");
        assertEquals("test_features", knn.getFeaturesCol());
        assertEquals("test_label", knn.getLabelCol());
        assertEquals(4, (int) knn.getK());
        assertEquals("test_prediction", knn.getPredictionCol());
    }

    @Test
    public void testOutputSchema() throws Exception {
        Knn knn =
                new Knn()
                        .setLabelCol("test_label")
                        .setFeaturesCol("test_features")
                        .setK(4)
                        .setPredictionCol("test_prediction");
        KnnModel model = knn.fit(trainData.as("test_features, test_label"));
        Table output = model.transform(predictData.as("test_features, test_label"))[0];
        assertEquals(
                Arrays.asList("test_features", "test_label", "test_prediction"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFewerDistinctPointsThanCluster() throws Exception {
        Knn knn = new Knn();
        KnnModel model = knn.fit(predictData);
        Table output = model.transform(predictData)[0];
        verifyPredictionResult(output, knn.getLabelCol(), knn.getPredictionCol());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        Knn knn = new Knn();
        KnnModel knnModel = knn.fit(trainData);
        Table output = knnModel.transform(predictData)[0];
        verifyPredictionResult(output, knn.getLabelCol(), knn.getPredictionCol());
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        trainData = TestUtils.convertDataTypesToSparseInt(tEnv, trainData);
        predictData = TestUtils.convertDataTypesToSparseInt(tEnv, predictData);
        assertArrayEquals(
                new Class<?>[] {SparseVector.class, Integer.class},
                TestUtils.getColumnDataTypes(trainData));
        assertArrayEquals(
                new Class<?>[] {SparseVector.class, Integer.class},
                TestUtils.getColumnDataTypes(predictData));

        Knn knn = new Knn();
        KnnModel knnModel = knn.fit(trainData);
        Table output = knnModel.transform(predictData)[0];
        verifyPredictionResult(output, knn.getLabelCol(), knn.getPredictionCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        Knn knn = new Knn();
        Knn loadedKnn =
                TestUtils.saveAndReload(
                        tEnv, knn, tempFolder.newFolder().getAbsolutePath(), Knn::load);
        KnnModel knnModel = loadedKnn.fit(trainData);
        knnModel =
                TestUtils.saveAndReload(
                        tEnv, knnModel, tempFolder.newFolder().getAbsolutePath(), KnnModel::load);
        assertEquals(
                Arrays.asList("packedFeatures", "featureNormSquares", "labels"),
                knnModel.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = knnModel.transform(predictData)[0];
        verifyPredictionResult(output, knn.getLabelCol(), knn.getPredictionCol());
    }

    @Test
    public void testModelSaveLoadAndPredict() throws Exception {
        Knn knn = new Knn();
        KnnModel knnModel = knn.fit(trainData);
        KnnModel newModel =
                TestUtils.saveAndReload(
                        tEnv, knnModel, tempFolder.newFolder().getAbsolutePath(), KnnModel::load);
        Table output = newModel.transform(predictData)[0];
        verifyPredictionResult(output, knn.getLabelCol(), knn.getPredictionCol());
    }

    @Test
    public void testGetModelData() throws Exception {
        Knn knn = new Knn();
        KnnModel knnModel = knn.fit(trainData);
        Table modelData = knnModel.getModelData()[0];
        DataStream<Row> output = tEnv.toDataStream(modelData);
        assertEquals("packedFeatures", modelData.getResolvedSchema().getColumnNames().get(0));
        assertEquals("featureNormSquares", modelData.getResolvedSchema().getColumnNames().get(1));
        assertEquals("labels", modelData.getResolvedSchema().getColumnNames().get(2));
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        KnnModelData data =
                new KnnModelData(
                        (DenseMatrix) modelRows.get(0).getField(0),
                        (DenseVector) modelRows.get(0).getField(1),
                        (DenseVector) modelRows.get(0).getField(2));
        Assert.assertNotNull(data);
        assertEquals(2, data.packedFeatures.numRows());
        assertEquals(data.packedFeatures.numCols(), data.labels.size());
        assertEquals(data.featureNormSquares.size(), data.labels.size());
    }

    @Test
    public void testSetModelData() throws Exception {
        Knn knn = new Knn();
        KnnModel modelA = knn.fit(trainData);
        Table modelData = modelA.getModelData()[0];
        KnnModel modelB = new KnnModel().setModelData(modelData);
        ParamUtils.updateExistingParams(modelB, modelA.getParamMap());
        Table output = modelB.transform(predictData)[0];
        verifyPredictionResult(output, knn.getLabelCol(), knn.getPredictionCol());
    }
}
