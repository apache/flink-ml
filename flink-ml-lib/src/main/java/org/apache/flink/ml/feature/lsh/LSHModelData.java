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

package org.apache.flink.ml.feature.lsh;

import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;

/**
 * Base class for LSH model data. A concrete class extending this base class should implement how to
 * map a feature vector to multiple hash vectors, and how to calculate corresponding distance
 * between two feature vectors.
 */
abstract class LSHModelData {
    /**
     * Maps an input feature vector to multiple hash vectors.
     *
     * @param vec input vector.
     * @return the mapping of LSH functions.
     */
    public abstract DenseIntDoubleVector[] hashFunction(IntDoubleVector vec);

    /**
     * Calculates the distance between two different feature vectors using the corresponding
     * distance metric.
     *
     * @param x One input vector in the metric space.
     * @param y One input vector in the metric space.
     * @return The distance between x and y.
     */
    public abstract double keyDistance(IntDoubleVector x, IntDoubleVector y);
}
