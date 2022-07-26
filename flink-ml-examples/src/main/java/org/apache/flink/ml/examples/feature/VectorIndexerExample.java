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

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.vectorindexer.VectorIndexer;
import org.apache.flink.ml.feature.vectorindexer.VectorIndexerModel;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import java.util.Arrays;
import java.util.List;

/** Simple program that creates a VectorIndexer instance and uses it for feature engineering. */
public class VectorIndexerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        List<Row> trainInput =
                Arrays.asList(
                        Row.of(Vectors.dense(1, 1)),
                        Row.of(Vectors.dense(2, -1)),
                        Row.of(Vectors.dense(3, 1)),
                        Row.of(Vectors.dense(4, 0)),
                        Row.of(Vectors.dense(5, 0)));

        List<Row> predictInput =
                Arrays.asList(
                        Row.of(Vectors.dense(0, 2)),
                        Row.of(Vectors.dense(0, 0)),
                        Row.of(Vectors.dense(0, -1)));

        Table trainTable = tEnv.fromDataStream(env.fromCollection(trainInput)).as("input");
        Table predictTable = tEnv.fromDataStream(env.fromCollection(predictInput)).as("input");

        // Creates a VectorIndexer object and initializes its parameters.
        VectorIndexer vectorIndexer =
                new VectorIndexer()
                        .setInputCol("input")
                        .setOutputCol("output")
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID)
                        .setMaxCategories(3);

        // Trains the VectorIndexer Model.
        VectorIndexerModel model = vectorIndexer.fit(trainTable);

        // Uses the VectorIndexer Model for predictions.
        Table outputTable = model.transform(predictTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            System.out.printf(
                    "Input Value: %s \tOutput Value: %s\n",
                    row.getField(vectorIndexer.getInputCol()),
                    row.getField(vectorIndexer.getOutputCol()));
        }
    }
}
