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
import org.apache.flink.ml.linalg.Vectors;

import org.junit.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests {@link CosineDistanceMeasure}, {@link EuclideanDistanceMeasure} and {@link
 * ManhattanDistanceMeasure}.
 */
public class DistanceMeasureTest {
    private static final VectorWithNorm VECTOR_WITH_NORM_A =
            new VectorWithNorm(Vectors.sparse(3, new int[] {1, 2}, new double[] {1, 2}));
    private static final VectorWithNorm VECTOR_WITH_NORM_B =
            new VectorWithNorm(Vectors.dense(1, 2, 3));
    private static final VectorWithNorm[] CENTROIDS =
            new VectorWithNorm[] {
                new VectorWithNorm(Vectors.dense(0, 1, 2)),
                new VectorWithNorm(Vectors.dense(1, 2, 3)),
                new VectorWithNorm(Vectors.dense(2, 3, 4))
            };

    private static final double TOLERANCE = 1e-7;

    @Test
    public void testEuclidean() {
        DistanceMeasure distanceMeasure = EuclideanDistanceMeasure.getInstance();
        assertEquals(
                Math.sqrt(3),
                distanceMeasure.distance(VECTOR_WITH_NORM_A, VECTOR_WITH_NORM_B),
                TOLERANCE);
        assertEquals(0, distanceMeasure.findClosest(CENTROIDS, VECTOR_WITH_NORM_A));
        assertEquals(1, distanceMeasure.findClosest(CENTROIDS, VECTOR_WITH_NORM_B));
    }

    @Test
    public void testEuclideanOfIdenticalVectors() {
        VectorWithNorm vector = new VectorWithNorm(Vectors.dense(3.0, 3.0));
        DistanceMeasure distanceMeasure = EuclideanDistanceMeasure.getInstance();
        assertEquals(0, distanceMeasure.distance(vector, vector));
    }

    @Test
    public void testManhattan() {
        DistanceMeasure distanceMeasure = ManhattanDistanceMeasure.getInstance();
        assertEquals(
                3, distanceMeasure.distance(VECTOR_WITH_NORM_A, VECTOR_WITH_NORM_B), TOLERANCE);
        assertEquals(0, distanceMeasure.findClosest(CENTROIDS, VECTOR_WITH_NORM_A));
        assertEquals(1, distanceMeasure.findClosest(CENTROIDS, VECTOR_WITH_NORM_B));
    }

    @Test
    public void testCosine() {
        DistanceMeasure distanceMeasure = CosineDistanceMeasure.getInstance();
        assertEquals(
                0.04381711,
                distanceMeasure.distance(VECTOR_WITH_NORM_A, VECTOR_WITH_NORM_B),
                TOLERANCE);
        assertEquals(0, distanceMeasure.findClosest(CENTROIDS, VECTOR_WITH_NORM_A));
        assertEquals(1, distanceMeasure.findClosest(CENTROIDS, VECTOR_WITH_NORM_B));
    }
}
