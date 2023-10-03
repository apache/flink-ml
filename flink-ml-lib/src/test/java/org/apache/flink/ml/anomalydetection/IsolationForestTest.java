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

package org.apache.flink.ml.anomalydetection;

import org.apache.flink.ml.anomalydetection.isolationforest.IForest;
import org.apache.flink.ml.anomalydetection.isolationforest.IsolationForest;
import org.apache.flink.ml.anomalydetection.isolationforest.IsolationForestModel;
import org.apache.flink.ml.anomalydetection.isolationforest.IsolationForestModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Tests {@link IsolationForest} and {@link IsolationForestModel}. */
public class IsolationForestTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    private static final List<DenseVector> DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Vectors.dense(4),
                            Vectors.dense(1),
                            Vectors.dense(4),
                            Vectors.dense(5),
                            Vectors.dense(3),
                            Vectors.dense(6),
                            Vectors.dense(2),
                            Vectors.dense(5),
                            Vectors.dense(6),
                            Vectors.dense(2),
                            Vectors.dense(5),
                            Vectors.dense(7),
                            Vectors.dense(1),
                            Vectors.dense(8),
                            Vectors.dense(15),
                            Vectors.dense(33),
                            Vectors.dense(4),
                            Vectors.dense(7),
                            Vectors.dense(6),
                            Vectors.dense(7),
                            Vectors.dense(8),
                            Vectors.dense(55)));

    private static final List<Set<DenseVector>> expectedGroups =
            Arrays.asList(
                    new HashSet<>(
                            Arrays.asList(
                                    Vectors.dense(4),
                                    Vectors.dense(1),
                                    Vectors.dense(4),
                                    Vectors.dense(5),
                                    Vectors.dense(3),
                                    Vectors.dense(6),
                                    Vectors.dense(2),
                                    Vectors.dense(5),
                                    Vectors.dense(6),
                                    Vectors.dense(2),
                                    Vectors.dense(5),
                                    Vectors.dense(7),
                                    Vectors.dense(1),
                                    Vectors.dense(8),
                                    Vectors.dense(4),
                                    Vectors.dense(7),
                                    Vectors.dense(6),
                                    Vectors.dense(7),
                                    Vectors.dense(8))),
                    new HashSet<>(
                            Arrays.asList(
                                    Vectors.dense(15), Vectors.dense(33), Vectors.dense(55))));

    private static final double TOLERANCE = 1e-7;

    private Table dataTable;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        // Allow KryoSerializer Fallback.
        env.getConfig().enableGenericTypes();
        tEnv = StreamTableEnvironment.create(env);
        dataTable = tEnv.fromDataStream(env.fromCollection(DATA)).as("features");
    }

    /**
     * Aggregates feature by predictions. Results are returned as a list of sets, where elements in
     * the same set are features whose prediction results are the same.
     *
     * @param rows A list of rows containing feature and prediction columns
     * @param featuresCol Name of the column in the table that contains the features
     * @param predictionCol Name of the column in the table that contains the prediction result
     * @return A map containing the collected results
     */
    protected static List<Set<DenseVector>> groupFeaturesByPrediction(
            List<Row> rows, String featuresCol, String predictionCol) {
        Map<Integer, Set<DenseVector>> map = new HashMap<>();
        for (Row row : rows) {
            DenseVector vector = ((Vector) row.getField(featuresCol)).toDense();
            int predict = (Integer) row.getField(predictionCol);
            map.putIfAbsent(predict, new HashSet<>());
            map.get(predict).add(vector);
        }
        return new ArrayList<>(map.values());
    }

    @Test
    public void testParam() {
        IsolationForest isolationForest = new IsolationForest();
        assertEquals("features", isolationForest.getFeaturesCol());
        assertEquals("prediction", isolationForest.getPredictionCol());
        assertEquals(256, isolationForest.getMaxSamples());
        assertEquals(1.0, isolationForest.getMaxFeatures(), TOLERANCE);
        assertEquals(100, isolationForest.getNumTrees());

        isolationForest
                .setFeaturesCol("test_features")
                .setPredictionCol("test_prediction")
                .setMaxSamples(128)
                .setMaxFeatures(0.5)
                .setNumTrees(90);

        assertEquals("test_features", isolationForest.getFeaturesCol());
        assertEquals("test_prediction", isolationForest.getPredictionCol());
        assertEquals(128, isolationForest.getMaxSamples());
        assertEquals(0.5, isolationForest.getMaxFeatures(), TOLERANCE);
        assertEquals(90, isolationForest.getNumTrees());
    }

    @Test
    public void testOutputSchema() throws Exception {
        Table input = dataTable.as("test_feature");
        IsolationForest isolationForest =
                new IsolationForest()
                        .setFeaturesCol("test_feature")
                        .setPredictionCol("test_prediction");
        IsolationForestModel model = isolationForest.fit(input);
        Table output = model.transform(input)[0];

        assertEquals(
                Arrays.asList("test_feature", "test_prediction"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        IsolationForest isolationForest =
                new IsolationForest().setMaxSamples(256).setMaxFeatures(1.0).setNumTrees(100);

        IsolationForestModel model = isolationForest.fit(dataTable);
        Table output = model.transform(dataTable)[0];

        assertEquals(
                Arrays.asList("features", "prediction"),
                output.getResolvedSchema().getColumnNames());
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results,
                        isolationForest.getFeaturesCol(),
                        isolationForest.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        IsolationForest isolationForest =
                new IsolationForest().setMaxSamples(256).setMaxFeatures(1.0).setNumTrees(100);
        IsolationForest loadedIsolationForest =
                TestUtils.saveAndReload(
                        tEnv,
                        isolationForest,
                        tempFolder.newFolder().getAbsolutePath(),
                        IsolationForest::load);

        IsolationForestModel model = loadedIsolationForest.fit(dataTable);
        IsolationForestModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        IsolationForestModel::load);
        Table output = loadedModel.transform(dataTable)[0];
        assertEquals(
                Arrays.asList("iForest"),
                loadedModel.getModelData()[0].getResolvedSchema().getColumnNames());
        assertEquals(
                Arrays.asList("features", "prediction"),
                output.getResolvedSchema().getColumnNames());

        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results,
                        isolationForest.getFeaturesCol(),
                        isolationForest.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }

    @Test
    public void testGetModelData() throws Exception {
        IsolationForest isolationForest =
                new IsolationForest().setMaxSamples(256).setMaxFeatures(1.0).setNumTrees(100);
        IsolationForestModel model = isolationForest.fit(dataTable);
        assertEquals(
                Arrays.asList("iForest"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        DataStream<IsolationForestModelData> modelData =
                IsolationForestModelData.getModelDataStream(model.getModelData()[0]);
        List<IsolationForestModelData> collectedModelData =
                IteratorUtils.toList(modelData.executeAndCollect());

        IForest iForest = collectedModelData.get(0).iForest;

        if (iForest.center0 < 0.5) {
            throw new Exception("Predicted value and actual value are different.");
        }

        if (iForest.center1 > 0.5) {
            throw new Exception("Predicted value and actual value are different.");
        }
    }

    @Test
    public void testSetModelData() throws Exception {
        IsolationForest isolationForest =
                new IsolationForest().setMaxSamples(256).setMaxFeatures(1.0).setNumTrees(100);
        IsolationForestModel modelA = isolationForest.fit(dataTable);
        IsolationForestModel modelB =
                new IsolationForestModel().setModelData(modelA.getModelData());
        ParamUtils.updateExistingParams(modelB, modelA.getParamMap());

        Table output = modelB.transform(dataTable)[0];
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results,
                        isolationForest.getFeaturesCol(),
                        isolationForest.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }
}
