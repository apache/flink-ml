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

import org.apache.flink.ml.feature.robustscaler.RobustScaler;
import org.apache.flink.ml.feature.robustscaler.RobustScalerModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a {@link RobustScaler} model and uses it for feature selection. */
public class RobustScalerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(1, Vectors.dense(0.0, 0.0)),
                        Row.of(2, Vectors.dense(1.0, -1.0)),
                        Row.of(3, Vectors.dense(2.0, -2.0)),
                        Row.of(4, Vectors.dense(3.0, -3.0)),
                        Row.of(5, Vectors.dense(4.0, -4.0)),
                        Row.of(6, Vectors.dense(5.0, -5.0)),
                        Row.of(7, Vectors.dense(6.0, -6.0)),
                        Row.of(8, Vectors.dense(7.0, -7.0)),
                        Row.of(9, Vectors.dense(8.0, -8.0)));
        Table trainTable = tEnv.fromDataStream(trainStream).as("id", "input");

        // Creates a RobustScaler object and initializes its parameters.
        RobustScaler robustScaler =
                new RobustScaler()
                        .setLower(0.25)
                        .setUpper(0.75)
                        .setRelativeError(0.001)
                        .setWithScaling(true)
                        .setWithCentering(true);

        // Trains the RobustScaler model.
        RobustScalerModel model = robustScaler.fit(trainTable);

        // Uses the RobustScaler model for predictions.
        Table outputTable = model.transform(trainTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector inputValue =
                    (DenseIntDoubleVector) row.getField(robustScaler.getInputCol());
            DenseIntDoubleVector outputValue =
                    (DenseIntDoubleVector) row.getField(robustScaler.getOutputCol());
            System.out.printf("Input Value: %-15s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
