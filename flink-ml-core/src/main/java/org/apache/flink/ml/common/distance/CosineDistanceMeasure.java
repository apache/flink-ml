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
import org.apache.flink.util.Preconditions;

/** Cosine distance between two vectors. */
public class CosineDistanceMeasure implements DistanceMeasure {

    private static final CosineDistanceMeasure instance = new CosineDistanceMeasure();
    public static final String NAME = "cosine";

    private CosineDistanceMeasure() {}

    public static CosineDistanceMeasure getInstance() {
        return instance;
    }

    @Override
    public double distance(VectorWithNorm v1, VectorWithNorm v2) {
        Preconditions.checkArgument(
                v1.l2Norm > 0 && v2.l2Norm > 0,
                "Consine distance is not defined for zero-length vectors.");
        return 1 - BLAS.dot(v1.vector, v2.vector) / v1.l2Norm / v2.l2Norm;
    }
}
