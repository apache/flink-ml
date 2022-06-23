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

import org.apache.flink.ml.feature.stringindexer.IndexToStringModel;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModelData;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/**
 * Simple program that creates a IndexToStringModelExample instance and uses it for feature
 * engineering.
 */
public class IndexToStringModelExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Creates model data for IndexToStringModel.
        StringIndexerModelData modelData =
                new StringIndexerModelData(
                        new String[][] {{"a", "b", "c", "d"}, {"-1.0", "0.0", "1.0", "2.0"}});
        Table modelTable = tEnv.fromDataStream(env.fromElements(modelData)).as("stringArrays");

        // Generates input data.
        DataStream<Row> predictStream = env.fromElements(Row.of(0, 3), Row.of(1, 2));
        Table predictTable = tEnv.fromDataStream(predictStream).as("inputCol1", "inputCol2");

        // Creates an indexToStringModel object and initializes its parameters.
        IndexToStringModel indexToStringModel =
                new IndexToStringModel()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setModelData(modelTable);

        // Uses the indexToStringModel object for feature transformations.
        Table outputTable = indexToStringModel.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            int[] inputValues = new int[indexToStringModel.getInputCols().length];
            String[] outputValues = new String[indexToStringModel.getInputCols().length];
            for (int i = 0; i < inputValues.length; i++) {
                inputValues[i] = (int) row.getField(indexToStringModel.getInputCols()[i]);
                outputValues[i] = (String) row.getField(indexToStringModel.getOutputCols()[i]);
            }

            System.out.printf(
                    "Input Values: %s \tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}
