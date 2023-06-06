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

package org.apache.flink.ml.examples.regression;

import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.regression.linearregression.LinearRegression;
import org.apache.flink.ml.regression.linearregression.LinearRegressionModel;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a LinearRegression model and uses it for regression. */
public class LinearRegressionExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(2, 1), 4.0, 1.0),
                        Row.of(Vectors.dense(3, 2), 7.0, 1.0),
                        Row.of(Vectors.dense(4, 3), 10.0, 1.0),
                        Row.of(Vectors.dense(2, 4), 10.0, 1.0),
                        Row.of(Vectors.dense(2, 2), 6.0, 1.0),
                        Row.of(Vectors.dense(4, 3), 10.0, 1.0),
                        Row.of(Vectors.dense(1, 2), 5.0, 1.0),
                        Row.of(Vectors.dense(5, 3), 11.0, 1.0));
        Table inputTable = tEnv.fromDataStream(inputStream).as("features", "label", "weight");

        // Creates a LinearRegression object and initializes its parameters.
        LinearRegression lr = new LinearRegression().setWeightCol("weight");

        // Trains the LinearRegression Model.
        LinearRegressionModel lrModel = lr.fit(inputTable);

        // Uses the LinearRegression Model for predictions.
        Table outputTable = lrModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector features =
                    (DenseIntDoubleVector) row.getField(lr.getFeaturesCol());
            double expectedResult = (Double) row.getField(lr.getLabelCol());
            double predictionResult = (Double) row.getField(lr.getPredictionCol());
            System.out.printf(
                    "Features: %s \tExpected Result: %s \tPrediction Result: %s\n",
                    features, expectedResult, predictionResult);
        }
    }
}
