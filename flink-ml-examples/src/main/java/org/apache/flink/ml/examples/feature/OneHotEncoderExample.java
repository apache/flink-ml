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

import org.apache.flink.ml.feature.onehotencoder.OneHotEncoder;
import org.apache.flink.ml.feature.onehotencoder.OneHotEncoderModel;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a OneHotEncoder model and uses it for feature engineering. */
public class OneHotEncoderExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(Row.of(0.0), Row.of(1.0), Row.of(2.0), Row.of(0.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("input");

        DataStream<Row> predictStream = env.fromElements(Row.of(0.0), Row.of(1.0), Row.of(2.0));
        Table predictTable = tEnv.fromDataStream(predictStream).as("input");

        // Creates a OneHotEncoder object and initializes its parameters.
        OneHotEncoder oneHotEncoder =
                new OneHotEncoder().setInputCols("input").setOutputCols("output");

        // Trains the OneHotEncoder Model.
        OneHotEncoderModel model = oneHotEncoder.fit(trainTable);

        // Uses the OneHotEncoder Model for predictions.
        Table outputTable = model.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            Double inputValue = (Double) row.getField(oneHotEncoder.getInputCols()[0]);
            SparseIntDoubleVector outputValue =
                    (SparseIntDoubleVector) row.getField(oneHotEncoder.getOutputCols()[0]);
            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
