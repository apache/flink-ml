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

package org.apache.flink.ml.examples;

import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example of how to use Flink ML that initializes and trains a KMeans model, and uses it to predict
 * the cluster id of evaluation data set. Note, since KMeans is an unsupervised learning model, the
 * dataset will not contain a label column.
 */
public class KMeansExample {

    private static final Logger LOG = LoggerFactory.getLogger(KMeansExample.class);

    public static void main(String[] args) {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        String featuresColumn = "features";
        String predictionColumn = "prediction";

        // Generate train data as DataStream.
        DataStream<DenseVector> trainingInputStream =
                env.fromElements(
                        Vectors.dense(0.0, 0.0),
                        Vectors.dense(0.0, 0.3),
                        Vectors.dense(0.2, 0.2),
                        Vectors.dense(0.3, 0.0),
                        Vectors.dense(0.3, 0.3),
                        Vectors.dense(9.0, 0.0),
                        Vectors.dense(9.0, 0.6),
                        Vectors.dense(9.2, 0.3),
                        Vectors.dense(9.6, 0.0),
                        Vectors.dense(9.6, 0.6));

        // Generate evaluation data as DataStream for prediction.
        DataStream<DenseVector> evaluationInputStream =
                env.fromElements(
                        Vectors.dense(0.0, 0.1),
                        Vectors.dense(0.0, 0.2),
                        Vectors.dense(0.1, 0.1),
                        Vectors.dense(0.2, 0.3),
                        Vectors.dense(0.5, 0.5),
                        Vectors.dense(9.0, 0.2),
                        Vectors.dense(9.2, 0.5),
                        Vectors.dense(9.4, 0.1),
                        Vectors.dense(9.6, 0.4),
                        Vectors.dense(9.8, 0.8));

        // Convert the training data from DataStream to Table, as Flink ML uses Table API.
        Table trainingData = tEnv.fromDataStream(trainingInputStream).as(featuresColumn);

        // Convert the evaluation data from DataStream to Table, as Flink ML uses Table API.
        Table evaluationData = tEnv.fromDataStream(evaluationInputStream).as(featuresColumn);

        // Creates a K-means object and initialize its parameters.
        KMeans kmeans =
                new KMeans()
                        .setK(2)
                        .setSeed(1L)
                        .setFeaturesCol(featuresColumn)
                        .setPredictionCol(predictionColumn);

        // Trains the K-means Model.
        KMeansModel model = kmeans.fit(trainingData);

        // Use the K-means Model for predictions.
        Table output = model.transform(evaluationData)[0];

        // Extracts and displays prediction result.
        for (CloseableIterator<Row> it = output.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector vector = (DenseVector) row.getField(featuresColumn);
            int clusterId = (Integer) row.getField(predictionColumn);
            LOG.info("Vector: {} \tCluster ID: {}", vector, clusterId);
        }
    }
}
