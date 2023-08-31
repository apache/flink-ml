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

package org.apache.flink.ml.examples.anomalydetection;

import org.apache.flink.ml.anomalydetection.isolationforest.IsolationForest;
import org.apache.flink.ml.anomalydetection.isolationforest.IsolationForestModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates an IsolationForest instance and uses it for anomaly detection. */
public class IsolationForestExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(10);
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<DenseVector> inputStream =
                env.fromElements(
                        Vectors.dense(1, 2),
                        Vectors.dense(1.1, 2),
                        Vectors.dense(1, 2.1),
                        Vectors.dense(1.1, 2.1),
                        Vectors.dense(0.1, 0.1));

        DataStream<DenseVector> inputStream2 =
                env.fromElements(
                        Vectors.dense(4),
                        Vectors.dense(1),
                        Vectors.dense(4),
                        Vectors.dense(5),
                        Vectors.dense(3),
                        Vectors.dense(6),
                        Vectors.dense(2),
                        Vectors.dense(5),
                        Vectors.dense(6),
                        Vectors.dense(2),
                        Vectors.dense(5),
                        Vectors.dense(7),
                        Vectors.dense(1),
                        Vectors.dense(8),
                        Vectors.dense(12),
                        Vectors.dense(33),
                        Vectors.dense(4),
                        Vectors.dense(7),
                        Vectors.dense(6),
                        Vectors.dense(7),
                        Vectors.dense(8),
                        Vectors.dense(55));

        Table inputTable = tEnv.fromDataStream(inputStream).as("features");

        IsolationForest isolationForest = new IsolationForest().setTreesNumber(100).setIters(1);

        IsolationForestModel isolationForestModel = isolationForest.fit(inputTable);

        Table outputTable = isolationForestModel.transform(inputTable)[0];

        // Extracts and displays the results.
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseVector features = (DenseVector) row.getField(isolationForest.getFeaturesCol());
            int predictId = (Integer) row.getField(isolationForest.getPredictionCol());
            System.out.printf("Features: %s \tPrediction: %s\n", features, predictId);
        }
    }
}
