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

package org.apache.flink.ml.examples.classification;

import org.apache.flink.ml.classification.knn.Knn;
import org.apache.flink.ml.classification.knn.KnnModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a Knn model and uses it for classification. */
public class KnnExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
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
                        Row.of(Vectors.dense(2.1, 3.1), 1.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("features", "label");

        DataStream<Row> predictStream =
                env.fromElements(
                        Row.of(Vectors.dense(4.0, 4.1), 5.0), Row.of(Vectors.dense(300, 42), 2.0));
        Table predictTable = tEnv.fromDataStream(predictStream).as("features", "label");

        // Creates a Knn object and initializes its parameters.
        Knn knn = new Knn().setK(4);

        // Trains the Knn Model.
        KnnModel knnModel = knn.fit(trainTable);

        // Uses the Knn Model for predictions.
        Table outputTable = knnModel.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector features =
                    (DenseIntDoubleVector) row.getField(knn.getFeaturesCol());
            double expectedResult = (Double) row.getField(knn.getLabelCol());
            double predictionResult = (Double) row.getField(knn.getPredictionCol());
            System.out.printf(
                    "Features: %-15s \tExpected Result: %s \tPrediction Result: %s\n",
                    features, expectedResult, predictionResult);
        }
    }
}
