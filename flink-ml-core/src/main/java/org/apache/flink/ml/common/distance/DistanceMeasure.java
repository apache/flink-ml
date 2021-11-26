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

import org.apache.flink.ml.linalg.Vector;

import java.io.Serializable;

/** Interface for measuring distance between two vectors. */
public interface DistanceMeasure extends Serializable {

    static DistanceMeasure getInstance(String distanceMeasure) {
        if (distanceMeasure.equals(EuclideanDistanceMeasure.NAME)) {
            return EuclideanDistanceMeasure.getInstance();
        }
        throw new IllegalArgumentException(
                "distanceMeasure "
                        + distanceMeasure
                        + " is not recognized. Supported options: 'euclidean'.");
    }

    /**
     * Measures the distance between two vectors.
     *
     * <p>Required: The two vectors should have the same dimension.
     */
    double distance(Vector v1, Vector v2);
}
