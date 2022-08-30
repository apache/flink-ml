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

import org.apache.flink.ml.feature.binarizer.Binarizer;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a Binarizer instance and uses it for feature engineering. */
public class BinarizerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(
                                1,
                                Vectors.dense(1, 2),
                                Vectors.sparse(
                                        17, new int[] {0, 3, 9}, new double[] {1.0, 2.0, 7.0})),
                        Row.of(
                                2,
                                Vectors.dense(2, 1),
                                Vectors.sparse(
                                        17, new int[] {0, 2, 14}, new double[] {5.0, 4.0, 1.0})),
                        Row.of(
                                3,
                                Vectors.dense(5, 18),
                                Vectors.sparse(
                                        17, new int[] {0, 11, 12}, new double[] {2.0, 4.0, 4.0})));

        Table inputTable = tEnv.fromDataStream(inputStream).as("f0", "f1", "f2");

        // Creates a Binarizer object and initializes its parameters.
        Binarizer binarizer =
                new Binarizer()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCols("of0", "of1", "of2")
                        .setThresholds(0.0, 0.0, 0.0);

        // Transforms input data.
        Table outputTable = binarizer.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            Object[] inputValues = new Object[binarizer.getInputCols().length];
            Object[] outputValues = new Object[binarizer.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = row.getField(binarizer.getInputCols()[i]);
                outputValues[i] = row.getField(binarizer.getOutputCols()[i]);
            }

            System.out.printf(
                    "Input Values: %s\tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}
