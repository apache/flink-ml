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

import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizer;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerModel;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerParams;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that trains a KBinsDiscretizer model and uses it for feature engineering. */
public class KBinsDiscretizerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(Vectors.dense(1, 10, 0)),
                        Row.of(Vectors.dense(1, 10, 0)),
                        Row.of(Vectors.dense(1, 10, 0)),
                        Row.of(Vectors.dense(4, 10, 0)),
                        Row.of(Vectors.dense(5, 10, 0)),
                        Row.of(Vectors.dense(6, 10, 0)),
                        Row.of(Vectors.dense(7, 10, 0)),
                        Row.of(Vectors.dense(10, 10, 0)),
                        Row.of(Vectors.dense(13, 10, 3)));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input");

        // Creates a KBinsDiscretizer object and initializes its parameters.
        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer().setNumBins(3).setStrategy(KBinsDiscretizerParams.UNIFORM);

        // Trains the KBinsDiscretizer Model.
        KBinsDiscretizerModel model = kBinsDiscretizer.fit(inputTable);

        // Uses the KBinsDiscretizer Model for predictions.
        Table outputTable = model.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector inputValue =
                    (DenseIntDoubleVector) row.getField(kBinsDiscretizer.getInputCol());
            DenseIntDoubleVector outputValue =
                    (DenseIntDoubleVector) row.getField(kBinsDiscretizer.getOutputCol());
            System.out.printf("Input Value: %s\tOutput Value: %s\n", inputValue, outputValue);
        }
    }
}
