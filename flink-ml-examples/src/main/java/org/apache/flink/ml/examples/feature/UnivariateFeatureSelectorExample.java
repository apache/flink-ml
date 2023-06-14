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

import org.apache.flink.ml.feature.univariatefeatureselector.UnivariateFeatureSelector;
import org.apache.flink.ml.feature.univariatefeatureselector.UnivariateFeatureSelectorModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that trains a {@link UnivariateFeatureSelector} model and uses it for feature
 * selection.
 */
public class UnivariateFeatureSelectorExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input training and prediction data.
        DataStream<Row> trainStream =
                env.fromElements(
                        Row.of(Vectors.dense(1.7, 4.4, 7.6, 5.8, 9.6, 2.3), 3.0),
                        Row.of(Vectors.dense(8.8, 7.3, 5.7, 7.3, 2.2, 4.1), 2.0),
                        Row.of(Vectors.dense(1.2, 9.5, 2.5, 3.1, 8.7, 2.5), 1.0),
                        Row.of(Vectors.dense(3.7, 9.2, 6.1, 4.1, 7.5, 3.8), 2.0),
                        Row.of(Vectors.dense(8.9, 5.2, 7.8, 8.3, 5.2, 3.0), 4.0),
                        Row.of(Vectors.dense(7.9, 8.5, 9.2, 4.0, 9.4, 2.1), 4.0));
        Table trainTable = tEnv.fromDataStream(trainStream).as("features", "label");

        // Creates a UnivariateFeatureSelector object and initializes its parameters.
        UnivariateFeatureSelector univariateFeatureSelector =
                new UnivariateFeatureSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setFeatureType("continuous")
                        .setLabelType("categorical")
                        .setSelectionThreshold(1);

        // Trains the UnivariateFeatureSelector model.
        UnivariateFeatureSelectorModel model = univariateFeatureSelector.fit(trainTable);

        // Uses the UnivariateFeatureSelector model for predictions.
        Table outputTable = model.transform(trainTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector inputValue =
                    (DenseIntDoubleVector) row.getField(univariateFeatureSelector.getFeaturesCol());
            DenseIntDoubleVector outputValue =
                    (DenseIntDoubleVector) row.getField(univariateFeatureSelector.getOutputCol());
            System.out.printf("Input Value: %-15s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
