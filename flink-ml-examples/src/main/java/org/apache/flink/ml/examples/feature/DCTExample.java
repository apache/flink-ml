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

import org.apache.flink.ml.feature.dct.DCT;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

/** Simple program that creates a DCT instance and uses it for feature engineering. */
public class DCTExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        List<IntDoubleVector> inputData =
                Arrays.asList(
                        Vectors.dense(1.0, 1.0, 1.0, 1.0), Vectors.dense(1.0, 0.0, -1.0, 0.0));
        Table inputTable = tEnv.fromDataStream(env.fromCollection(inputData)).as("input");

        // Creates a DCT object and initializes its parameters.
        DCT dct = new DCT();

        // Uses the DCT object for feature transformations.
        Table outputTable = dct.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            IntDoubleVector inputValue = row.getFieldAs(dct.getInputCol());
            IntDoubleVector outputValue = row.getFieldAs(dct.getOutputCol());

            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
