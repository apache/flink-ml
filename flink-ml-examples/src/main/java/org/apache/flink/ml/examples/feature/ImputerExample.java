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

import org.apache.flink.ml.feature.imputer.Imputer;
import org.apache.flink.ml.feature.imputer.ImputerModel;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that trains a {@link Imputer} model and uses it for feature engineering. */
public class ImputerExample {

    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Double.NaN, 9.0),
                        Row.of(1.0, 9.0),
                        Row.of(1.5, 9.0),
                        Row.of(2.5, Double.NaN),
                        Row.of(5.0, 5.0),
                        Row.of(5.0, 4.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("input1", "input2");

        // Creates an Imputer object and initialize its parameters
        Imputer imputer =
                new Imputer()
                        .setInputCols("input1", "input2")
                        .setOutputCols("output1", "output2")
                        .setStrategy("mean")
                        .setMissingValue(Double.NaN);

        // Trains the Imputer model.
        ImputerModel model = imputer.fit(trainTable);

        // Uses the Imputer model for predictions.
        Table outputTable = model.transform(trainTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            double[] inputValues = new double[imputer.getInputCols().length];
            double[] outputValues = new double[imputer.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = (double) row.getField(imputer.getInputCols()[i]);
                outputValues[i] = (double) row.getField(imputer.getOutputCols()[i]);
            }
            System.out.printf(
                    "Input Values: %s\tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}
