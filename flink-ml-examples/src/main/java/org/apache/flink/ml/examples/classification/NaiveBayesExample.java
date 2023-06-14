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

import org.apache.flink.ml.classification.naivebayes.NaiveBayes;
import org.apache.flink.ml.classification.naivebayes.NaiveBayesModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a NaiveBayes model and uses it for classification. */
public class NaiveBayesExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Vectors.dense(0, 0.), 11),
                        Row.of(Vectors.dense(1, 0), 10),
                        Row.of(Vectors.dense(1, 1.), 10));
        Table trainTable = tEnv.fromDataStream(trainStream).as("features", "label");

        DataStream<Row> predictStream =
                env.fromElements(
                        Row.of(Vectors.dense(0, 1.)),
                        Row.of(Vectors.dense(0, 0.)),
                        Row.of(Vectors.dense(1, 0)),
                        Row.of(Vectors.dense(1, 1.)));
        Table predictTable = tEnv.fromDataStream(predictStream).as("features");

        // Creates a NaiveBayes object and initializes its parameters.
        NaiveBayes naiveBayes =
                new NaiveBayes()
                        .setSmoothing(1.0)
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setModelType("multinomial");

        // Trains the NaiveBayes Model.
        NaiveBayesModel naiveBayesModel = naiveBayes.fit(trainTable);

        // Uses the NaiveBayes Model for predictions.
        Table outputTable = naiveBayesModel.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector features =
                    (DenseIntDoubleVector) row.getField(naiveBayes.getFeaturesCol());
            double predictionResult = (Double) row.getField(naiveBayes.getPredictionCol());
            System.out.printf("Features: %s \tPrediction Result: %s\n", features, predictionResult);
        }
    }
}
