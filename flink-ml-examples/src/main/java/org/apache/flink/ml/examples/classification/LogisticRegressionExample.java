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

import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a LogisticRegression model and uses it for classification. */
public class LogisticRegressionExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(1, 2, 3, 4), 0., 1.),
                        Row.of(Vectors.dense(2, 2, 3, 4), 0., 2.),
                        Row.of(Vectors.dense(3, 2, 3, 4), 0., 3.),
                        Row.of(Vectors.dense(4, 2, 3, 4), 0., 4.),
                        Row.of(Vectors.dense(5, 2, 3, 4), 0., 5.),
                        Row.of(Vectors.dense(11, 2, 3, 4), 1., 1.),
                        Row.of(Vectors.dense(12, 2, 3, 4), 1., 2.),
                        Row.of(Vectors.dense(13, 2, 3, 4), 1., 3.),
                        Row.of(Vectors.dense(14, 2, 3, 4), 1., 4.),
                        Row.of(Vectors.dense(15, 2, 3, 4), 1., 5.));
        Table inputTable = tEnv.fromDataStream(inputStream).as("features", "label", "weight");

        // Creates a LogisticRegression object and initializes its parameters.
        LogisticRegression lr = new LogisticRegression().setWeightCol("weight");

        // Trains the LogisticRegression Model.
        LogisticRegressionModel lrModel = lr.fit(inputTable);

        // Uses the LogisticRegression Model for predictions.
        Table outputTable = lrModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector features =
                    (DenseIntDoubleVector) row.getField(lr.getFeaturesCol());
            double expectedResult = (Double) row.getField(lr.getLabelCol());
            double predictionResult = (Double) row.getField(lr.getPredictionCol());
            DenseIntDoubleVector rawPredictionResult =
                    (DenseIntDoubleVector) row.getField(lr.getRawPredictionCol());
            System.out.printf(
                    "Features: %-25s \tExpected Result: %s \tPrediction Result: %s \tRaw Prediction Result: %s\n",
                    features, expectedResult, predictionResult, rawPredictionResult);
        }
    }
}
