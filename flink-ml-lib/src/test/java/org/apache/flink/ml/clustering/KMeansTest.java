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

package org.apache.flink.ml.clustering;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.clustering.kmeans.KMeansModelData;
import org.apache.flink.ml.common.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Tests {@link KMeans} and {@link KMeansModel}. */
public class KMeansTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private static final List<DenseVector> DATA =
            Arrays.asList(
                    Vectors.dense(0.0, 0.0),
                    Vectors.dense(0.0, 0.3),
                    Vectors.dense(0.3, 0.0),
                    Vectors.dense(9.0, 0.0),
                    Vectors.dense(9.0, 0.6),
                    Vectors.dense(9.6, 0.0));
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private static final List<Set<DenseVector>> expectedGroups =
            Arrays.asList(
                    new HashSet<>(
                            Arrays.asList(
                                    Vectors.dense(0.0, 0.0),
                                    Vectors.dense(0.0, 0.3),
                                    Vectors.dense(0.3, 0.0))),
                    new HashSet<>(
                            Arrays.asList(
                                    Vectors.dense(9.0, 0.0),
                                    Vectors.dense(9.0, 0.6),
                                    Vectors.dense(9.6, 0.0))));
    private Table dataTable;

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
        KMeans kmeans = new KMeans();
        assertEquals("features", kmeans.getFeaturesCol());
        assertEquals("prediction", kmeans.getPredictionCol());
        assertEquals(EuclideanDistanceMeasure.NAME, kmeans.getDistanceMeasure());
        assertEquals("random", kmeans.getInitMode());
        assertEquals(2, kmeans.getK());
        assertEquals(20, kmeans.getMaxIter());
        assertEquals(KMeans.class.getName().hashCode(), kmeans.getSeed());

        kmeans.setK(9)
                .setFeaturesCol("test_feature")
                .setPredictionCol("test_prediction")
                .setK(3)
                .setMaxIter(30)
                .setSeed(100);

        assertEquals("test_feature", kmeans.getFeaturesCol());
        assertEquals("test_prediction", kmeans.getPredictionCol());
        assertEquals(3, kmeans.getK());
        assertEquals(30, kmeans.getMaxIter());
        assertEquals(100, kmeans.getSeed());
    }

    @Test
    public void testOutputSchema() {
        Table input = dataTable.as("test_feature");
        KMeans kmeans =
                new KMeans().setFeaturesCol("test_feature").setPredictionCol("test_prediction");
        KMeansModel model = kmeans.fit(input);
        Table output = model.transform(input)[0];

        assertEquals(
                Arrays.asList("test_feature", "test_prediction"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFewerDistinctPointsThanCluster() {
        List<DenseVector> data =
                Arrays.asList(
                        Vectors.dense(0.0, 0.1), Vectors.dense(0.0, 0.1), Vectors.dense(0.0, 0.1));

        Table input = tEnv.fromDataStream(env.fromCollection(data)).as("features");

        KMeans kmeans = new KMeans().setK(2);
        KMeansModel model = kmeans.fit(input);
        Table output = model.transform(input)[0];

        List<Set<DenseVector>> expectedGroups =
                Collections.singletonList(Collections.singleton(Vectors.dense(0.0, 0.1)));
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }

    @Test
    public void testFitAndPredict() {
        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel model = kmeans.fit(dataTable);
        Table output = model.transform(dataTable)[0];

        assertEquals(
                Arrays.asList("features", "prediction"),
                output.getResolvedSchema().getColumnNames());
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }

    @Test
    public void testInputTypeConversion() {
        dataTable = TestUtils.convertDataTypesToSparseInt(tEnv, dataTable);
        assertArrayEquals(
                new Class<?>[] {SparseVector.class}, TestUtils.getColumnDataTypes(dataTable));

        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel model = kmeans.fit(dataTable);
        Table output = model.transform(dataTable)[0];

        assertEquals(
                Arrays.asList("features", "prediction"),
                output.getResolvedSchema().getColumnNames());
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeans loadedKmeans =
                TestUtils.saveAndReload(tEnv, kmeans, tempFolder.newFolder().getAbsolutePath());
        KMeansModel model = loadedKmeans.fit(dataTable);
        KMeansModel loadedModel =
                TestUtils.saveAndReload(tEnv, model, tempFolder.newFolder().getAbsolutePath());
        Table output = loadedModel.transform(dataTable)[0];
        assertEquals(
                Arrays.asList("centroids", "weights"),
                loadedModel.getModelData()[0].getResolvedSchema().getColumnNames());
        assertEquals(
                Arrays.asList("features", "prediction"),
                output.getResolvedSchema().getColumnNames());

        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }

    @Test
    public void testGetModelData() throws Exception {
        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel model = kmeans.fit(dataTable);
        assertEquals(
                Arrays.asList("centroids", "weights"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        DataStream<KMeansModelData> modelData =
                KMeansModelData.getModelDataStream(model.getModelData()[0]);
        List<KMeansModelData> collectedModelData =
                IteratorUtils.toList(modelData.executeAndCollect());
        assertEquals(1, collectedModelData.size());
        DenseVector[] centroids = collectedModelData.get(0).centroids;
        assertEquals(2, centroids.length);
        Arrays.sort(centroids, Comparator.comparingDouble(vector -> vector.get(0)));
        assertArrayEquals(centroids[0].values, new double[] {0.1, 0.1}, 1e-5);
        assertArrayEquals(centroids[1].values, new double[] {9.2, 0.2}, 1e-5);
    }

    @Test
    public void testSetModelData() {
        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel modelA = kmeans.fit(dataTable);
        KMeansModel modelB = new KMeansModel().setModelData(modelA.getModelData());
        ReadWriteUtils.updateExistingParams(modelB, modelA.getParamMap());

        Table output = modelB.transform(dataTable)[0];
        List<Row> results = IteratorUtils.toList(output.execute().collect());
        List<Set<DenseVector>> actualGroups =
                groupFeaturesByPrediction(
                        results, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        assertTrue(CollectionUtils.isEqualCollection(expectedGroups, actualGroups));
    }
}
