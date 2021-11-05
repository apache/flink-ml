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

package org.apache.flink.ml.clustering;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

import java.util.LinkedList;
import java.util.List;

/** Provides the default data sets used for the k-means tests. */
public class KMeansData {
    public static final double[][] POINTS =
            new double[][] {
                new double[] {-14.22, -48.01},
                new double[] {-22.78, 37.10},
                new double[] {56.18, -42.99},
                new double[] {35.04, 50.29},
                new double[] {-9.53, -46.26},
                new double[] {-34.35, 48.25},
                new double[] {55.82, -57.49},
                new double[] {21.03, 54.64},
                new double[] {-13.63, -42.26},
                new double[] {-36.57, 32.63},
                new double[] {50.65, -52.40},
                new double[] {24.48, 34.04},
                new double[] {-2.69, -36.02},
                new double[] {-38.80, 36.58},
                new double[] {24.00, -53.74},
                new double[] {32.41, 24.96},
                new double[] {-4.32, -56.92},
                new double[] {-22.68, 29.42},
                new double[] {59.02, -39.56},
                new double[] {24.47, 45.07},
                new double[] {5.23, -41.20},
                new double[] {-23.00, 38.15},
                new double[] {44.55, -51.50},
                new double[] {14.62, 59.06},
                new double[] {7.41, -56.05},
                new double[] {-26.63, 28.97},
                new double[] {47.37, -44.72},
                new double[] {29.07, 51.06},
                new double[] {0.59, -31.89},
                new double[] {-39.09, 20.78},
                new double[] {42.97, -48.98},
                new double[] {34.36, 49.08},
                new double[] {-21.91, -49.01},
                new double[] {-46.68, 46.04},
                new double[] {48.52, -43.67},
                new double[] {30.05, 49.25},
                new double[] {4.03, -43.56},
                new double[] {-37.85, 41.72},
                new double[] {38.24, -48.32},
                new double[] {20.83, 57.85}
            };

    public static Table getData(StreamExecutionEnvironment env) {
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        List<DenseVector> features = new LinkedList<>();
        for (double[] point : POINTS) {
            features.add(new DenseVector(point));
        }
        Schema schema = Schema.newBuilder().column("f0", DataTypes.of(DenseVector.class)).build();
        return tEnv.fromDataStream(env.fromCollection(features), schema).as("features");
    }
}
