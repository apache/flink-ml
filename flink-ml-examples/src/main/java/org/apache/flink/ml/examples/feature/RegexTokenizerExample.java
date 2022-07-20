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

import org.apache.flink.ml.feature.regextokenizer.RegexTokenizer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;

/** Simple program that creates a RegexTokenizer instance and uses it for feature engineering. */
public class RegexTokenizerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(Row.of("Test for tokenization."), Row.of("Te,st. punct"));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a RegexTokenizer object and initializes its parameters.
        RegexTokenizer regexTokenizer =
                new RegexTokenizer()
                        .setInputCol("input")
                        .setOutputCol("output")
                        .setPattern("\\w+|\\p{Punct}");

        // Uses the Tokenizer object for feature transformations.
        Table outputTable = regexTokenizer.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String inputValue = (String) row.getField(regexTokenizer.getInputCol());
            String[] outputValues = (String[]) row.getField(regexTokenizer.getOutputCol());

            System.out.printf(
                    "Input Value: %s \tOutput Values: %s\n",
                    inputValue, Arrays.toString(outputValues));
        }
    }
}
