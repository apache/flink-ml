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

package org.apache.flink.ml.benchmark;

import java.lang.reflect.Field;
import java.util.Map;

/** Wraps all possible parameters for ML benchmarks in a single class. */
public class BenchmarkParams {
    public long randomSeed = 2022L;
    public long numExamples = 100;
    public long numTestExamples = 10;
    public int numPartitions = 3;
    public int numFeatures = 10;
    public double tol = 0.1;
    public int maxIter = 10;
    public int k = 5;
    public int globalBatchSize = 100;
    public double learningRate = 0.1;
    public double reg = 0.01;

    public BenchmarkParams() {}

    public static BenchmarkParams fromParams(Map<String, Object> params) {
        BenchmarkParams benchmarkParams = new BenchmarkParams();
        for (Map.Entry<String, Object> nameAndVal : params.entrySet()) {
            try {
                Field field = benchmarkParams.getClass().getField(nameAndVal.getKey());
                field.set(benchmarkParams, nameAndVal.getValue());
            } catch (NoSuchFieldException | IllegalAccessException ignored) {
            }
        }
        return benchmarkParams;
    }

    @Override
    public String toString() {
        return "BenchParams{"
                + "randomSeed="
                + randomSeed
                + ", numExamples="
                + numExamples
                + ", numTestExamples="
                + numTestExamples
                + ", numPartitions="
                + numPartitions
                + ", numFeatures="
                + numFeatures
                + ", tol="
                + tol
                + ", maxIter="
                + maxIter
                + ", k="
                + k
                + ", globalBatchSize="
                + globalBatchSize
                + ", learningRate="
                + learningRate
                + ", reg="
                + reg
                + '}';
    }
}
