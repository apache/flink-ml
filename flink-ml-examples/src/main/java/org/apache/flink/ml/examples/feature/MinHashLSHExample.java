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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.feature.lsh.MinHashLSH;
import org.apache.flink.ml.feature.lsh.MinHashLSHModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;

import java.util.Arrays;
import java.util.List;

import static org.apache.flink.table.api.Expressions.$;

/**
 * Simple program that trains a MinHashLSH model and uses it for approximate nearest neighbors and
 * similarity join.
 */
public class MinHashLSHExample {
    public static void main(String[] args) throws Exception {

        // Creates a new StreamExecutionEnvironment.
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Creates a StreamTableEnvironment.
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates two datasets.
        Table dataA =
                tEnv.fromDataStream(
                        env.fromCollection(
                                Arrays.asList(
                                        Row.of(
                                                0,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {0, 1, 2},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                1,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {2, 3, 4},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                2,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {0, 2, 4},
                                                        new double[] {1., 1., 1.}))),
                                Types.ROW_NAMED(
                                        new String[] {"id", "vec"},
                                        Types.INT,
                                        TypeInformation.of(SparseIntDoubleVector.class))));

        Table dataB =
                tEnv.fromDataStream(
                        env.fromCollection(
                                Arrays.asList(
                                        Row.of(
                                                3,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {1, 3, 5},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                4,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {2, 3, 5},
                                                        new double[] {1., 1., 1.})),
                                        Row.of(
                                                5,
                                                Vectors.sparse(
                                                        6,
                                                        new int[] {1, 2, 4},
                                                        new double[] {1., 1., 1.}))),
                                Types.ROW_NAMED(
                                        new String[] {"id", "vec"},
                                        Types.INT,
                                        TypeInformation.of(SparseIntDoubleVector.class))));

        // Creates a MinHashLSH estimator object and initializes its parameters.
        MinHashLSH lsh =
                new MinHashLSH()
                        .setInputCol("vec")
                        .setOutputCol("hashes")
                        .setSeed(2022)
                        .setNumHashTables(5);

        // Trains the MinHashLSH model.
        MinHashLSHModel model = lsh.fit(dataA);

        // Uses the MinHashLSH model for transformation.
        Table output = model.transform(dataA)[0];

        // Extracts and displays the results.
        List<String> fieldNames = output.getResolvedSchema().getColumnNames();
        for (Row result :
                (List<Row>) IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect())) {
            IntDoubleVector inputValue = result.getFieldAs(fieldNames.indexOf(lsh.getInputCol()));
            DenseIntDoubleVector[] outputValue =
                    result.getFieldAs(fieldNames.indexOf(lsh.getOutputCol()));
            System.out.printf(
                    "Vector: %s \tHash values: %s\n", inputValue, Arrays.toString(outputValue));
        }

        // Finds approximate nearest neighbors of the key.
        IntDoubleVector key = Vectors.sparse(6, new int[] {1, 3}, new double[] {1., 1.});
        output = model.approxNearestNeighbors(dataA, key, 2).select($("id"), $("distCol"));
        for (Row result :
                (List<Row>) IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect())) {
            int idValue = result.getFieldAs(fieldNames.indexOf("id"));
            double distValue = result.getFieldAs(result.getArity() - 1);
            System.out.printf("ID: %d \tDistance: %f\n", idValue, distValue);
        }

        // Approximately finds pairs from two datasets with distances smaller than the threshold.
        output = model.approxSimilarityJoin(dataA, dataB, .6, "id");
        for (Row result :
                (List<Row>) IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect())) {
            int idAValue = result.getFieldAs(0);
            int idBValue = result.getFieldAs(1);
            double distValue = result.getFieldAs(2);
            System.out.printf(
                    "ID from left: %d \tID from right: %d \t Distance: %f\n",
                    idAValue, idBValue, distValue);
        }
    }
}
