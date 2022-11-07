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

import org.apache.flink.ml.feature.randomsplitter.RandomSplitter;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates a RandomSplitter instance and uses it for data splitting. */
public class RandomSplitterExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(1, 10, 0),
                        Row.of(1, 10, 0),
                        Row.of(1, 10, 0),
                        Row.of(4, 10, 0),
                        Row.of(5, 10, 0),
                        Row.of(6, 10, 0),
                        Row.of(7, 10, 0),
                        Row.of(10, 10, 0),
                        Row.of(13, 10, 3));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a RandomSplitter object and initializes its parameters.
        RandomSplitter splitter = new RandomSplitter().setWeights(4.0, 6.0);

        // Uses the RandomSplitter to split inputData.
        Table[] outputTable = splitter.transform(inputTable);

        // Extracts and displays the results.
        System.out.println("Split Result 1 (40%)");
        for (CloseableIterator<Row> it = outputTable[0].execute().collect(); it.hasNext(); ) {
            System.out.printf("%s\n", it.next());
        }
        System.out.println("Split Result 2 (60%)");
        for (CloseableIterator<Row> it = outputTable[1].execute().collect(); it.hasNext(); ) {
            System.out.printf("%s\n", it.next());
        }
    }
}
