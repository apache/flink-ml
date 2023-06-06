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

package org.apache.flink.ml.examples.clustering;

import org.apache.flink.ml.clustering.agglomerativeclustering.AgglomerativeClustering;
import org.apache.flink.ml.clustering.agglomerativeclustering.AgglomerativeClusteringParams;
import org.apache.flink.ml.common.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

/** Simple program that creates an AgglomerativeClustering instance and uses it for clustering. */
public class AgglomerativeClusteringExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<DenseIntDoubleVector> inputStream =
                env.fromElements(
                        Vectors.dense(1, 1),
                        Vectors.dense(1, 4),
                        Vectors.dense(1, 0),
                        Vectors.dense(4, 1.5),
                        Vectors.dense(4, 4),
                        Vectors.dense(4, 0));
        Table inputTable = tEnv.fromDataStream(inputStream).as("features");

        // Creates an AgglomerativeClustering object and initializes its parameters.
        AgglomerativeClustering agglomerativeClustering =
                new AgglomerativeClustering()
                        .setLinkage(AgglomerativeClusteringParams.LINKAGE_WARD)
                        .setDistanceMeasure(EuclideanDistanceMeasure.NAME)
                        .setPredictionCol("prediction")
                        .setComputeFullTree(true);

        // Uses the AgglomerativeClustering object for clustering.
        Table[] outputs = agglomerativeClustering.transform(inputTable);

        // Extracts and displays the clustering results.
        for (CloseableIterator<Row> it = outputs[0].execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            DenseIntDoubleVector features =
                    (DenseIntDoubleVector) row.getField(agglomerativeClustering.getFeaturesCol());
            int clusterId = (Integer) row.getField(agglomerativeClustering.getPredictionCol());
            System.out.printf("Features: %s \tCluster ID: %s\n", features, clusterId);
        }
    }
}
