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

package org.apache.flink.ml.examples.recommendation;

import org.apache.flink.ml.recommendation.swing.Swing;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that creates a Swing instance and uses it to generate recommendations for items.
 */
public class SwingExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(0L, 10L),
                        Row.of(0L, 11L),
                        Row.of(0L, 12L),
                        Row.of(1L, 13L),
                        Row.of(1L, 12L),
                        Row.of(2L, 10L),
                        Row.of(2L, 11L),
                        Row.of(2L, 12L),
                        Row.of(3L, 13L),
                        Row.of(3L, 12L));

        Table inputTable = tEnv.fromDataStream(inputStream).as("user", "item");

        // Creates a Swing object and initializes its parameters.
        Swing swing = new Swing().setUserCol("user").setItemCol("item").setMinUserBehavior(1);

        // Transforms the data.
        Table[] outputTable = swing.transform(inputTable);

        // Extracts and displays the result of swing algorithm.
        for (CloseableIterator<Row> it = outputTable[0].execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            long mainItem = row.getFieldAs(0);
            String itemRankScore = row.getFieldAs(1);

            System.out.printf("item: %d, top-k similar items: %s\n", mainItem, itemRankScore);
        }
    }
}
