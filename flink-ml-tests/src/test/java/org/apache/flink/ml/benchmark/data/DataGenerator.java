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

package org.apache.flink.ml.benchmark.data;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.source.datagen.RandomGenerator;
import org.apache.flink.util.NumberSequenceIterator;

import java.util.Random;

/** Utility class to generate synthetic data set for benchmark machine learning algorithms. */
public class DataGenerator {

    /**
     * Generates data points with continuous features.
     *
     * @param env The execution environment instance.
     * @param numExamples Number of examples to generate in total.
     * @param seed The seed to generate seed on each partition.
     * @param numPartitions Number of partitions to generate.
     * @param numFeatures Number of features for each data point.
     * @return The generated data set for benchmark.
     */
    public static DataStream<Vector> generateContinuousFeatures(
            StreamExecutionEnvironment env,
            long numExamples,
            long seed,
            int numPartitions,
            int numFeatures) {
        int[] featureArity = new int[numFeatures];
        long[] seeds = new long[numPartitions];
        Random random = new Random(seed);
        for (int i = 0; i < numPartitions; i++) {
            seeds[i] = random.nextLong();
        }
        return env.fromParallelCollection(
                        new NumberSequenceIterator(1L, numExamples), BasicTypeInfo.LONG_TYPE_INFO)
                .map(new GenerateFeatureFunction(featureArity, seeds))
                .setParallelism(numPartitions);
    }

    private static class GenerateFeatureFunction extends RichMapFunction<Long, Vector> {
        private final long[] allSeeds;
        private final int[] featureArity;
        private FeatureGenerator featureGenerator;

        GenerateFeatureFunction(int[] featureArity, long[] allSeeds) {
            this.allSeeds = allSeeds;
            this.featureArity = featureArity;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            long seedOnThisPartition = allSeeds[getRuntimeContext().getIndexOfThisSubtask()];
            featureGenerator = new FeatureGenerator(featureArity, seedOnThisPartition);
        }

        @Override
        public Vector map(Long value) {
            return featureGenerator.next();
        }
    }

    /**
     * Generates features according to the given feature arity and the random seed.
     *
     * <p>Each element in the given feature arity represents number of possible categorical values
     * of that dimension. Zero indicates a continuous feature.
     */
    private static class FeatureGenerator extends RandomGenerator<Vector> {
        final int[] featureArity;
        final Random random;

        public FeatureGenerator(int[] featureArity, long seed) {
            this.featureArity = featureArity;
            random = new Random(seed);
        }

        @Override
        public Vector next() {
            double[] features = new double[featureArity.length];
            for (int i = 0; i < features.length; i++) {
                if (featureArity[i] == 0) {
                    features[i] = 2 * random.nextDouble() - 1;
                } else {
                    features[i] = random.nextInt(featureArity[i]);
                }
            }
            return Vectors.dense(features);
        }
    }
}
