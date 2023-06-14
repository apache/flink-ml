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

package org.apache.flink.ml.examples.feature;

import org.apache.flink.ml.feature.maxabsscaler.MaxAbsScaler;
import org.apache.flink.ml.feature.maxabsscaler.MaxAbsScalerModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a MaxAbsScaler model and uses it for feature engineering. */
public class MaxAbsScalerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Vectors.dense(0.0, 3.0)),
                        Row.of(Vectors.dense(2.1, 0.0)),
                        Row.of(Vectors.dense(4.1, 5.1)),
                        Row.of(Vectors.dense(6.1, 8.1)),
                        Row.of(Vectors.dense(200, 400)));
        Table trainTable = tEnv.fromDataStream(trainStream).as("input");

        DataStream<Row> predictStream =
                env.fromElements(
                        Row.of(Vectors.dense(150.0, 90.0)),
                        Row.of(Vectors.dense(50.0, 40.0)),
                        Row.of(Vectors.dense(100.0, 50.0)));
        Table predictTable = tEnv.fromDataStream(predictStream).as("input");

        // Creates a MaxAbsScaler object and initializes its parameters.
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();

        // Trains the MaxAbsScaler Model.
        MaxAbsScalerModel maxAbsScalerModel = maxAbsScaler.fit(trainTable);

        // Uses the MaxAbsScaler Model for predictions.
        Table outputTable = maxAbsScalerModel.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector inputValue =
                    (DenseIntDoubleVector) row.getField(maxAbsScaler.getInputCol());
            DenseIntDoubleVector outputValue =
                    (DenseIntDoubleVector) row.getField(maxAbsScaler.getOutputCol());
            System.out.printf("Input Value: %-15s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
