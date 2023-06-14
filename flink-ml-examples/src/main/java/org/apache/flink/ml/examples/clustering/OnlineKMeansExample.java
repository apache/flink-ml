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

package org.apache.flink.ml.examples.clustering;

import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.clustering.kmeans.KMeansModelData;
import org.apache.flink.ml.clustering.kmeans.OnlineKMeans;
import org.apache.flink.ml.clustering.kmeans.OnlineKMeansModel;
import org.apache.flink.ml.examples.util.PeriodicSourceFunction;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Simple program that trains an OnlineKMeans model and uses it for clustering. */
public class OnlineKMeansExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data. Both are infinite streams that periodically
        // sends out provided data to trigger model update and prediction.
        List<Row> trainData1 =
                Arrays.asList(
                        Row.of(Vectors.dense(0.0, 0.0)),
                        Row.of(Vectors.dense(0.0, 0.3)),
                        Row.of(Vectors.dense(0.3, 0.0)),
                        Row.of(Vectors.dense(9.0, 0.0)),
                        Row.of(Vectors.dense(9.0, 0.6)),
                        Row.of(Vectors.dense(9.6, 0.0)));

        List<Row> trainData2 =
                Arrays.asList(
                        Row.of(Vectors.dense(10.0, 100.0)),
                        Row.of(Vectors.dense(10.0, 100.3)),
                        Row.of(Vectors.dense(10.3, 100.0)),
                        Row.of(Vectors.dense(-10.0, -100.0)),
                        Row.of(Vectors.dense(-10.0, -100.6)),
                        Row.of(Vectors.dense(-10.6, -100.0)));

        List<Row> predictData =
                Arrays.asList(
                        Row.of(Vectors.dense(10.0, 10.0)), Row.of(Vectors.dense(-10.0, 10.0)));

        SourceFunction<Row> trainSource =
                new PeriodicSourceFunction(1000, Arrays.asList(trainData1, trainData2));
        DataStream<Row> trainStream =
                env.addSource(trainSource, new RowTypeInfo(DenseIntDoubleVectorTypeInfo.INSTANCE));
        Table trainTable = tEnv.fromDataStream(trainStream).as("features");

        SourceFunction<Row> predictSource =
                new PeriodicSourceFunction(1000, Collections.singletonList(predictData));
        DataStream<Row> predictStream =
                env.addSource(
                        predictSource, new RowTypeInfo(DenseIntDoubleVectorTypeInfo.INSTANCE));
        Table predictTable = tEnv.fromDataStream(predictStream).as("features");

        // Creates an online K-means object and initializes its parameters and initial model data.
        OnlineKMeans onlineKMeans =
                new OnlineKMeans()
                        .setFeaturesCol("features")
                        .setPredictionCol("prediction")
                        .setGlobalBatchSize(6)
                        .setInitialModelData(
                                KMeansModelData.generateRandomModelData(tEnv, 2, 2, 0.0, 0));

        // Trains the online K-means Model.
        OnlineKMeansModel onlineModel = onlineKMeans.fit(trainTable);

        // Uses the online K-means Model for predictions.
        Table outputTable = onlineModel.transform(predictTable)[0];

        // Extracts and displays the results. As training data stream continuously triggers the
        // update of the internal k-means model data, clustering results of the same predict dataset
        // would change over time.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row1 = it.next();
            DenseIntDoubleVector features1 =
                    (DenseIntDoubleVector) row1.getField(onlineKMeans.getFeaturesCol());
            Integer clusterId1 = (Integer) row1.getField(onlineKMeans.getPredictionCol());
            Row row2 = it.next();
            DenseIntDoubleVector features2 =
                    (DenseIntDoubleVector) row2.getField(onlineKMeans.getFeaturesCol());
            Integer clusterId2 = (Integer) row2.getField(onlineKMeans.getPredictionCol());
            if (Objects.equals(clusterId1, clusterId2)) {
                System.out.printf("%s and %s are now in the same cluster.\n", features1, features2);
            } else {
                System.out.printf(
                        "%s and %s are now in different clusters.\n", features1, features2);
            }
        }
    }
}
