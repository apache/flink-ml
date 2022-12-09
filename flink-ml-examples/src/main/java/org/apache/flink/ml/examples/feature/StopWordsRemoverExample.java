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

import org.apache.flink.ml.feature.stopwordsremover.StopWordsRemover;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a StopWordsRemover instance and uses it for feature engineering. */
public class StopWordsRemoverExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of((Object) new String[] {"test", "test"}),
                        Row.of((Object) new String[] {"a", "b", "c", "d"}),
                        Row.of((Object) new String[] {"a", "the", "an"}),
                        Row.of((Object) new String[] {"A", "The", "AN"}),
                        Row.of((Object) new String[] {null}),
                        Row.of((Object) new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a StopWordsRemover object and initializes its parameters.
        StopWordsRemover remover =
                new StopWordsRemover().setInputCols("input").setOutputCols("output");

        // Uses the StopWordsRemover object for feature transformations.
        Table outputTable = remover.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String[] inputValues = row.getFieldAs("input");
            String[] outputValues = row.getFieldAs("output");

            System.out.printf(
                    "Input Values: %s\tOutput Values: %s\n",
                    Arrays.toString(inputValues), Arrays.toString(outputValues));
        }
    }
}
