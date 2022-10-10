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

package org.apache.flink.ml.examples.stats;

import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.stats.chisqtest.ChiSqTest;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates a ChiSqTest instance and uses it for statistics. */
public class ChiSqTestExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        Table inputTable =
                tEnv.fromDataStream(
                                env.fromElements(
                                        Row.of(0., Vectors.dense(5, 1.)),
                                        Row.of(2., Vectors.dense(6, 2.)),
                                        Row.of(1., Vectors.dense(7, 2.)),
                                        Row.of(1., Vectors.dense(5, 4.)),
                                        Row.of(0., Vectors.dense(5, 1.)),
                                        Row.of(2., Vectors.dense(6, 2.)),
                                        Row.of(1., Vectors.dense(7, 2.)),
                                        Row.of(1., Vectors.dense(5, 4.)),
                                        Row.of(2., Vectors.dense(5, 1.)),
                                        Row.of(0., Vectors.dense(5, 2.)),
                                        Row.of(0., Vectors.dense(5, 2.)),
                                        Row.of(1., Vectors.dense(9, 4.)),
                                        Row.of(1., Vectors.dense(9, 3.))))
                        .as("label", "features");

        // Creates a ChiSqTest object and initializes its parameters.
        ChiSqTest chiSqTest =
                new ChiSqTest().setFlatten(true).setFeaturesCol("features").setLabelCol("label");

        // Uses the ChiSqTest object for statistics.
        Table outputTable = chiSqTest.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            System.out.printf(
                    "Feature Index: %s\tP Value: %s\tDegree of Freedom: %s\tStatistics: %s\n",
                    row.getField("featureIndex"),
                    row.getField("pValue"),
                    row.getField("degreeOfFreedom"),
                    row.getField("statistic"));
        }
    }
}
