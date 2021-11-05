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
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.distance.DistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Test;

import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Tests KMeans and KMeansModel. */
public class KMeansTest extends AbstractTestBase {

    //    @Test
    public void testCheckpointWithMetadata() throws Exception {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);
        env.enableCheckpointing(100);

        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        Schema schema =
                Schema.newBuilder()
                        .column("f0", DataTypes.of(Integer.class))
                        .columnByMetadata("rowtime", "TIMESTAMP_LTZ(3)")
                        .build();

        Table data = tEnv.fromDataStream(env.fromCollection(Arrays.asList(1, 2, 3)), schema);
        tEnv.toDataStream(data).print();

        env.execute();
    }

    @Test
    public void testKMeansDataStream() throws Exception {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(1);
        env.enableCheckpointing(100);

        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        Table data = KMeansData.getData(env);
        KMeans kmeans = new KMeans().setMaxIter(10).setK(2);
        KMeansModel model = kmeans.fit(data);

        String tempDir = Files.createTempDirectory("").toString();
        model.save(tempDir);
        env.execute();

        model = KMeansModel.load(tempDir);

        Table output = model.transform(data)[0];

        DataStream<Tuple2<DenseVector, Integer>> pointsWithClusterId =
                tEnv.toDataStream(output)
                        .map(
                                new MapFunction<Row, Tuple2<DenseVector, Integer>>() {
                                    @Override
                                    public Tuple2<DenseVector, Integer> map(Row row) {
                                        return Tuple2.of(
                                                (DenseVector) row.getField(kmeans.getFeaturesCol()),
                                                (Integer) row.getField(kmeans.getPredictionCol()));
                                    }
                                });

        List<Tuple2<DenseVector, Integer>> pointsWithClusterIdList =
                IteratorUtils.toList(pointsWithClusterId.executeAndCollect());
        System.out.println("pointsWithClusterId size " + pointsWithClusterIdList.size());
        System.out.println("pointsWithClusterId " + pointsWithClusterIdList);

        Assert.assertEquals(pointsWithClusterIdList.size(), KMeansData.POINTS.length);
        pointsWithClusterIdList.stream()
                .map(t -> t.f1)
                .forEach(
                        clusterId ->
                                Assert.assertTrue(clusterId >= 0 && clusterId < kmeans.getK()));

        DistanceMeasure distanceMeasure = DistanceMeasure.getInstance(model.getDistanceMeasure());
        double loss = computeCost(pointsWithClusterIdList, distanceMeasure);
        System.out.println("Loss " + loss);
    }

    private static double computeCost(
            List<Tuple2<DenseVector, Integer>> pointsWithCentroids,
            DistanceMeasure distanceMeasure) {
        Map<Integer, List<DenseVector>> pointsByClusterId =
                pointsWithCentroids.stream()
                        .collect(
                                Collectors.groupingBy(
                                        t -> t.f1,
                                        Collectors.mapping(t -> t.f0, Collectors.toList())));

        System.out.println("pointsByClusterId " + pointsByClusterId);
        double loss = 0;
        for (Map.Entry<Integer, List<DenseVector>> entry : pointsByClusterId.entrySet()) {
            DenseVector meanPoint = getMeanPoint(entry.getValue());
            for (DenseVector point : entry.getValue()) {
                loss += distanceMeasure.distance(meanPoint, point);
            }
        }
        return loss;
    }

    private static DenseVector getMeanPoint(List<DenseVector> points) {
        int dim = points.get(0).size();

        DenseVector meanPoint = new DenseVector(new double[dim]);
        Arrays.fill(meanPoint.values, 0);

        for (int i = 0; i < points.size(); i++) {
            for (int j = 0; j < dim; j++) {
                meanPoint.values[j] += points.get(i).values[j];
            }
        }
        for (int i = 0; i < dim; i++) {
            meanPoint.values[i] /= points.size();
        }

        return meanPoint;
    }
}
