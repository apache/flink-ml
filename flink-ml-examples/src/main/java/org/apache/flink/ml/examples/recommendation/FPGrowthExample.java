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

import org.apache.flink.ml.recommendation.fpgrowth.FPGrowth;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/**
 * Simple program that creates a FPGrowth instance and uses it to generate frequent patterns and
 * association rules.
 */
public class FPGrowthExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(""),
                        Row.of("A,B,C,D"),
                        Row.of("B,C,E"),
                        Row.of("A,B,C,E"),
                        Row.of("B,D,E"),
                        Row.of("A,B,C,D,A"));

        Table inputTable = tEnv.fromDataStream(inputStream).as("items");

        // Creates a FPGrowth object and initializes its parameters.
        FPGrowth fpg = new FPGrowth().setMinSupportCount(3);

        // Transforms the data.
        Table[] outputTable = fpg.transform(inputTable);

        // Extracts and displays the frequent patterns.
        for (CloseableIterator<Row> it = outputTable[0].execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String pattern = row.getFieldAs(0);
            Long support = row.getFieldAs(1);
            Long itemCount = row.getFieldAs(2);

            System.out.printf(
                    "pattern: %s, support count: %d, item_count:%d\n", pattern, support, itemCount);
        }

        // Extracts and displays the association rules.
        for (CloseableIterator<Row> it = outputTable[1].execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String rule = row.getFieldAs(0);
            Double lift = row.getFieldAs(2);
            Double support = row.getFieldAs(3);
            Double confidence_percent = row.getFieldAs(4);

            System.out.printf(
                    "rule: %s, list: %f, support:%f, confidence:%f\n",
                    rule, lift, support, confidence_percent);
        }
    }
}
