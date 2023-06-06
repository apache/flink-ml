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

package org.apache.flink.ml.common.util;

import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;

import java.util.ArrayList;
import java.util.List;

/** Provides utility functions for {@link IntDoubleVector}. */
public class VectorUtils {
    /**
     * Selects a subset of the vector base on the indices. Note that the input indices must be
     * sorted in ascending order.
     */
    public static IntDoubleVector selectByIndices(IntDoubleVector vector, int[] sortedIndices) {
        if (vector instanceof DenseIntDoubleVector) {
            DenseIntDoubleVector resultVec = new DenseIntDoubleVector(sortedIndices.length);
            for (int i = 0; i < sortedIndices.length; i++) {
                resultVec.set(i, vector.get(sortedIndices[i]));
            }
            return resultVec;
        } else {
            List<Integer> resultIndices = new ArrayList<>();
            List<Double> resultValues = new ArrayList<>();

            int[] indices = ((SparseIntDoubleVector) vector).indices;
            for (int i = 0, j = 0; i < indices.length && j < sortedIndices.length; ) {
                if (indices[i] == sortedIndices[j]) {
                    resultIndices.add(j++);
                    resultValues.add(((SparseIntDoubleVector) vector).values[i++]);
                } else if (indices[i] > sortedIndices[j]) {
                    j++;
                } else {
                    i++;
                }
            }
            return new SparseIntDoubleVector(
                    sortedIndices.length,
                    resultIndices.stream().mapToInt(Integer::intValue).toArray(),
                    resultValues.stream().mapToDouble(Double::doubleValue).toArray());
        }
    }
}
