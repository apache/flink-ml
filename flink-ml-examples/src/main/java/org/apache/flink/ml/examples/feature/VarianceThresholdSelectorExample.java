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

import org.apache.flink.ml.feature.variancethresholdselector.VarianceThresholdSelector;
import org.apache.flink.ml.feature.variancethresholdselector.VarianceThresholdSelectorModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that trains a {@link VarianceThresholdSelector} model and uses it for feature
 * selection.
 */
public class VarianceThresholdSelectorExample {

    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(1, Vectors.dense(5.0, 7.0, 0.0, 7.0, 6.0, 0.0)),
                        Row.of(2, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0)),
                        Row.of(3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0)),
                        Row.of(4, Vectors.dense(1.0, 9.0, 8.0, 5.0, 7.0, 4.0)),
                        Row.of(5, Vectors.dense(9.0, 8.0, 6.0, 5.0, 4.0, 4.0)),
                        Row.of(6, Vectors.dense(6.0, 9.0, 7.0, 0.0, 2.0, 0.0)));
        Table trainTable = tEnv.fromDataStream(trainStream).as("id", "input");

        // Create a VarianceThresholdSelector object and initialize its parameters
        double threshold = 8.0;
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector()
                        .setVarianceThreshold(threshold)
                        .setInputCol("input");

        // Train the VarianceThresholdSelector model.
        VarianceThresholdSelectorModel model = varianceThresholdSelector.fit(trainTable);

        // Uses the VarianceThresholdSelector model for predictions.
        Table outputTable = model.transform(trainTable)[0];

        // Extracts and displays the results.
        System.out.printf("Variance Threshold: %s\n", threshold);
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector inputValue =
                    (DenseIntDoubleVector) row.getField(varianceThresholdSelector.getInputCol());
            DenseIntDoubleVector outputValue =
                    (DenseIntDoubleVector) row.getField(varianceThresholdSelector.getOutputCol());
            System.out.printf("Input Values: %-15s\tOutput Values: %s\n", inputValue, outputValue);
        }
    }
}
