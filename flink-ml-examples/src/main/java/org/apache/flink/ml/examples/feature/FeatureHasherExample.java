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

import org.apache.flink.ml.feature.featurehasher.FeatureHasher;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a FeatureHasher instance and uses it for feature engineering. */
public class FeatureHasherExample {
    public static void main(String[] args) {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> dataStream =
                env.fromCollection(
                        Arrays.asList(Row.of(0, "a", 1.0, true), Row.of(1, "c", 1.0, false)));
        Table inputDataTable = tEnv.fromDataStream(dataStream).as("id", "f0", "f1", "f2");

        // Creates a FeatureHasher object and initializes its parameters.
        FeatureHasher featureHash =
                new FeatureHasher()
                        .setInputCols("f0", "f1", "f2")
                        .setCategoricalCols("f0", "f2")
                        .setOutputCol("vec")
                        .setNumFeatures(1000);

        // Uses the FeatureHasher object for feature transformations.
        Table outputTable = featureHash.transform(inputDataTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            Object[] inputValues = new Object[featureHash.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = row.getField(featureHash.getInputCols()[i]);
            }
            IntDoubleVector outputValue =
                    (IntDoubleVector) row.getField(featureHash.getOutputCol());

            System.out.printf(
                    "Input Values: %s \tOutput Value: %s\n",
                    Arrays.toString(inputValues), outputValue);
        }
    }
}
