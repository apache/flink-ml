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

/** Interface for measuring the Euclidean distance between two vectors. */
public class EuclideanDistanceMeasure implements DistanceMeasure {

    private static final EuclideanDistanceMeasure instance = new EuclideanDistanceMeasure();
    public static final String NAME = "euclidean";

    private EuclideanDistanceMeasure() {}

    public static EuclideanDistanceMeasure getInstance() {
        return instance;
    }

    @Override
    public double distance(Vector v1, Vector v2) {
        double squaredDistance = 0.0;

        for (int i = 0; i < v1.size(); i++) {
            double diff = v1.get(i) - v2.get(i);
            squaredDistance += diff * diff;
        }
        return Math.sqrt(squaredDistance);
    }
}
