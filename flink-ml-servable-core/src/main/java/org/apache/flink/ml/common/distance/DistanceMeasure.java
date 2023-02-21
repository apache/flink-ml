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

import org.apache.flink.ml.linalg.VectorWithNorm;

import java.io.Serializable;

/** Interface for measuring distance between two vectors. */
public interface DistanceMeasure extends Serializable {

    static DistanceMeasure getInstance(String distanceMeasure) {
        switch (distanceMeasure) {
            case EuclideanDistanceMeasure.NAME:
                return EuclideanDistanceMeasure.getInstance();
            case ManhattanDistanceMeasure.NAME:
                return ManhattanDistanceMeasure.getInstance();
            case CosineDistanceMeasure.NAME:
                return CosineDistanceMeasure.getInstance();
            default:
                throw new IllegalArgumentException(
                        "distanceMeasure "
                                + distanceMeasure
                                + " is not recognized. Supported options: 'euclidean, manhattan, cosine'.");
        }
    }

    /**
     * Measures the distance between two vectors.
     *
     * <p>Required: The two vectors should have the same dimension.
     */
    double distance(VectorWithNorm v1, VectorWithNorm v2);

    /** Finds the index of the closest center to the given point. */
    default int findClosest(VectorWithNorm[] centroids, VectorWithNorm point) {
        int targetCentroidId = -1;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < centroids.length; i++) {
            double distance = distance(centroids[i], point);
            if (distance < minDistance) {
                minDistance = distance;
                targetCentroidId = i;
            }
        }
        return targetCentroidId;
    }
}
