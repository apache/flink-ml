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

import org.apache.flink.ml.feature.standardscaler.StandardScaler;
import org.apache.flink.ml.feature.standardscaler.StandardScalerModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a StandardScaler model and uses it for feature engineering. */
public class StandardScalerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(-2.5, 9, 1)),
                        Row.of(Vectors.dense(1.4, -5, 1)),
                        Row.of(Vectors.dense(2, -1, -2)));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a StandardScaler object and initializes its parameters.
        StandardScaler standardScaler = new StandardScaler();

        // Trains the StandardScaler Model.
        StandardScalerModel model = standardScaler.fit(inputTable);

        // Uses the StandardScaler Model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector inputValue =
                    (DenseIntDoubleVector) row.getField(standardScaler.getInputCol());
            DenseIntDoubleVector outputValue =
                    (DenseIntDoubleVector) row.getField(standardScaler.getOutputCol());
            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
