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

package org.apache.flink.ml.common.distance;

import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.VectorWithNorm;

/** Euclidean distance (also known as L2 distance) between two vectors. */
public class EuclideanDistanceMeasure implements DistanceMeasure {

    private static final EuclideanDistanceMeasure instance = new EuclideanDistanceMeasure();
    public static final String NAME = "euclidean";

    private EuclideanDistanceMeasure() {}

    public static EuclideanDistanceMeasure getInstance() {
        return instance;
    }

    @Override
    public double distance(VectorWithNorm v1, VectorWithNorm v2) {
        return Math.sqrt(distanceSquare(v1, v2));
    }

    private double distanceSquare(VectorWithNorm v1, VectorWithNorm v2) {
        // Computing the distance square between two vectors that are close enough might result in
        // a negative value, due to the loss of data accuracy. A Math.max helps to guarantee that
        // a non-negative value is returned.
        return Math.max(
                0.0,
                v1.l2Norm * v1.l2Norm
                        + v2.l2Norm * v2.l2Norm
                        - 2.0 * BLAS.dot(v1.vector, v2.vector));
    }

    @Override
    public int findClosest(VectorWithNorm[] centroids, VectorWithNorm point) {
        double bestL2DistanceSquare = Double.POSITIVE_INFINITY;
        int bestIndex = 0;
        for (int i = 0; i < centroids.length; i++) {
            VectorWithNorm centroid = centroids[i];

            double lowerBoundSqrt = point.l2Norm - centroid.l2Norm;
            double lowerBound = lowerBoundSqrt * lowerBoundSqrt;
            if (lowerBound >= bestL2DistanceSquare) {
                continue;
            }

            double l2DistanceSquare = distanceSquare(point, centroid);
            if (l2DistanceSquare < bestL2DistanceSquare) {
                bestL2DistanceSquare = l2DistanceSquare;
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
