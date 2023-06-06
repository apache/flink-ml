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

package org.apache.flink.ml.examples;

import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

import static org.apache.flink.ml.Functions.vectorToArray;
import static org.apache.flink.table.api.Expressions.$;

/** Simple program that converts a column of dense/sparse vectors into a column of double arrays. */
public class VectorToArrayExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input vector data.
        List<Vector> vectors =
                Arrays.asList(
                        Vectors.dense(0.0, 0.0),
                        Vectors.sparse(2, new int[] {1}, new double[] {1.0}));
        Table inputTable =
                tEnv.fromDataStream(env.fromCollection(vectors, VectorTypeInfo.INSTANCE))
                        .as("vector");

        // Converts each vector to a double array.
        Table outputTable = inputTable.select($("vector"), vectorToArray($("vector")).as("array"));

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            IntDoubleVector vector = row.getFieldAs("vector");
            Double[] doubleArray = row.getFieldAs("array");
            System.out.printf(
                    "Input vector: %s\tOutput double array: %s\n",
                    vector, Arrays.toString(doubleArray));
        }
    }
}
