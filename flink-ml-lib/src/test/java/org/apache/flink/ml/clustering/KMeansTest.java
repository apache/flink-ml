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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests KMeans and KMeansModel. */
public class KMeansTest extends AbstractTestBase {
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
    private Table dataTable;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);

        Schema schema = Schema.newBuilder().column("f0", DataTypes.of(DenseVector.class)).build();
        dataTable = tEnv.fromDataStream(env.fromCollection(DATA), schema).as("features");
    }

    // Executes the graph and returns a map which maps points to clusterId.
    private static Map<DenseVector, Integer> executeAndCollect(
            Table output, String featureCol, String predictionCol) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();

        DataStream<Tuple2<DenseVector, Integer>> stream =
                tEnv.toDataStream(output)
                        .map(
                                new MapFunction<Row, Tuple2<DenseVector, Integer>>() {
                                    @Override
                                    public Tuple2<DenseVector, Integer> map(Row row) {
                                        return Tuple2.of(
                                                (DenseVector) row.getField(featureCol),
                                                (Integer) row.getField(predictionCol));
                                    }
                                });

        List<Tuple2<DenseVector, Integer>> pointsWithClusterId =
                IteratorUtils.toList(stream.executeAndCollect());

        Map<DenseVector, Integer> clusterIdByPoints = new HashMap<>();
        for (Tuple2<DenseVector, Integer> entry : pointsWithClusterId) {
            clusterIdByPoints.put(entry.f0, entry.f1);
        }
        return clusterIdByPoints;
    }

    private static void verifyClusteringResult(
            Map<DenseVector, Integer> clusterIdByPoints, List<List<Integer>> groups) {
        for (List<Integer> group : groups) {
            for (int i = 1; i < group.size(); i++) {
                assertEquals(
                        clusterIdByPoints.get(DATA.get(group.get(0))),
                        clusterIdByPoints.get(DATA.get(group.get(i))));
            }
        }
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
    public void testFeaturePredictionParam() throws Exception {
        Table input = dataTable.as("test_feature");
        KMeans kmeans =
                new KMeans().setFeaturesCol("test_feature").setPredictionCol("test_prediction");
        KMeansModel model = kmeans.fit(input);
        Table output = model.transform(input)[0];

        assertEquals(
                Arrays.asList("test_feature", "test_prediction"),
                output.getResolvedSchema().getColumnNames());
        Map<DenseVector, Integer> clusterIdByPoints =
                executeAndCollect(output, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        verifyClusteringResult(
                clusterIdByPoints, Arrays.asList(Arrays.asList(0, 1, 2), Arrays.asList(3, 4, 5)));
    }

    @Test
    public void testFewerDistinctPointsThanCluster() throws Exception {
        List<DenseVector> data =
                Arrays.asList(
                        Vectors.dense(0.0, 0.1), Vectors.dense(0.0, 0.1), Vectors.dense(0.0, 0.1));

        Schema schema = Schema.newBuilder().column("f0", DataTypes.of(DenseVector.class)).build();
        Table input = tEnv.fromDataStream(env.fromCollection(data), schema).as("features");

        KMeans kmeans = new KMeans().setK(2);
        KMeansModel model = kmeans.fit(input);
        Table output = model.transform(input)[0];

        Map<DenseVector, Integer> clusterIdByPoints =
                executeAndCollect(output, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        assertEquals(Collections.singleton(0), new HashSet<>(clusterIdByPoints.values()));
    }

    @Test
    public void testFitAndPredict() throws Exception {
        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel model = kmeans.fit(dataTable);
        Table output = model.transform(dataTable)[0];

        assertEquals(
                Arrays.asList("features", "prediction"),
                output.getResolvedSchema().getColumnNames());
        Map<DenseVector, Integer> clusterIdByPoints =
                executeAndCollect(output, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        verifyClusteringResult(
                clusterIdByPoints, Arrays.asList(Arrays.asList(0, 1, 2), Arrays.asList(3, 4, 5)));
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        String path = Files.createTempDirectory("").toString();

        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel model = kmeans.fit(dataTable);

        model.save(path);
        env.execute();

        KMeansModel loadedModel = KMeansModel.load(env, path);
        Table output = loadedModel.transform(dataTable)[0];

        assertEquals(
                Arrays.asList("f0"),
                loadedModel.getModelData()[0].getResolvedSchema().getColumnNames());
        assertEquals(
                Arrays.asList("features", "prediction"),
                output.getResolvedSchema().getColumnNames());
        Map<DenseVector, Integer> clusterIdByPoints =
                executeAndCollect(output, kmeans.getFeaturesCol(), kmeans.getPredictionCol());
        verifyClusteringResult(
                clusterIdByPoints, Arrays.asList(Arrays.asList(0, 1, 2), Arrays.asList(3, 4, 5)));
    }

    @Test
    public void testGetModelData() throws Exception {
        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel modelA = kmeans.fit(dataTable);
        Table modelData = modelA.getModelData()[0];

        DataStream<DenseVector[]> output =
                tEnv.toDataStream(modelData).map(row -> (DenseVector[]) row.getField("f0"));

        assertEquals(Arrays.asList("f0"), modelData.getResolvedSchema().getColumnNames());
        List<DenseVector[]> centroids = IteratorUtils.toList(output.executeAndCollect());
        assertEquals(1, centroids.size());
        assertEquals(2, centroids.get(0).length);
        Arrays.sort(centroids.get(0), Comparator.comparingDouble(vector -> vector.get(0)));
        assertArrayEquals(centroids.get(0)[0].values, new double[] {0.1, 0.1}, 1e-5);
        assertArrayEquals(centroids.get(0)[1].values, new double[] {9.2, 0.2}, 1e-5);
    }

    @Test
    public void testSetModelData() throws Exception {
        KMeans kmeans = new KMeans().setMaxIter(2).setK(2);
        KMeansModel modelA = kmeans.fit(dataTable);
        Table modelData = modelA.getModelData()[0];

        KMeansModel modelB = new KMeansModel().setModelData(modelData);
        ReadWriteUtils.updateExistingParams(modelB, modelA.getParamMap());

        Table output = modelB.transform(dataTable)[0];
        Map<DenseVector, Integer> clusterIdByPoints =
                executeAndCollect(output, kmeans.getFeaturesCol(), kmeans.getPredictionCol());

        verifyClusteringResult(
                clusterIdByPoints, Arrays.asList(Arrays.asList(0, 1, 2), Arrays.asList(3, 4, 5)));
    }
}
