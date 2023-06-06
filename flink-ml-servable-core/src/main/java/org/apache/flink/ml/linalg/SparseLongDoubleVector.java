/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.flink.ml.linalg;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.linalg.typeinfo.SparseLongDoubleVectorTypeInfoFactory;
import org.apache.flink.util.Preconditions;

import java.util.Arrays;

/**
 * A sparse vector with long as keys and double as values.
 *
 * <p>TODO: Add processing logic for {@link SparseLongDoubleVector} for existing algorithms.
 */
@TypeInfo(SparseLongDoubleVectorTypeInfoFactory.class)
@PublicEvolving
public class SparseLongDoubleVector implements LongDoubleVector {

    public final long n;
    public long[] indices;
    public double[] values;

    public SparseLongDoubleVector(long n, long[] indices, double[] values) {
        this.n = n;
        this.indices = indices;
        this.values = values;
        if (!isIndicesSorted()) {
            sortIndices();
        }
        validateSortedData();
    }

    @Override
    public Long size() {
        return n;
    }

    @Override
    public Double get(Long i) {
        int pos = Arrays.binarySearch(indices, i);
        if (pos >= 0) {
            return values[pos];
        }
        return 0.;
    }

    @Override
    public void set(Long i, Double value) {
        int pos = Arrays.binarySearch(indices, i);
        if (pos >= 0) {
            values[pos] = value;
        } else if (value != 0.0) {
            Preconditions.checkArgument(i < n, "Index out of bounds: " + i);
            long[] indices = new long[this.indices.length + 1];
            double[] values = new double[this.indices.length + 1];
            System.arraycopy(this.indices, 0, indices, 0, -pos - 1);
            System.arraycopy(this.values, 0, values, 0, -pos - 1);
            indices[-pos - 1] = i;
            values[-pos - 1] = value;
            System.arraycopy(this.indices, -pos - 1, indices, -pos, this.indices.length + pos + 1);
            System.arraycopy(this.values, -pos - 1, values, -pos, this.indices.length + pos + 1);
            this.indices = indices;
            this.values = values;
        }
    }

    @Override
    public SparseLongDoubleVector toSparse() {
        return this;
    }

    /**
     * Checks whether input data is validate.
     *
     * <p>This function does the following checks:
     *
     * <ul>
     *   <li>The indices array and values array are of the same size.
     *   <li>vector indices are in valid range.
     *   <li>vector indices are unique.
     * </ul>
     *
     * <p>This function works as expected only when indices are sorted.
     */
    private void validateSortedData() {
        Preconditions.checkArgument(
                indices.length == values.length,
                "Indices size and values size should be the same.");
        if (this.indices.length > 0) {
            Preconditions.checkArgument(
                    this.indices[0] >= 0 && this.indices[this.indices.length - 1] < this.n,
                    "Index out of bound.");
        }
        for (int i = 1; i < this.indices.length; i++) {
            Preconditions.checkArgument(
                    this.indices[i] > this.indices[i - 1], "Indices duplicated.");
        }
    }

    private boolean isIndicesSorted() {
        for (int i = 1; i < this.indices.length; i++) {
            if (this.indices[i] < this.indices[i - 1]) {
                return false;
            }
        }
        return true;
    }

    /** Sorts the indices and values. */
    private void sortIndices() {
        sortImpl(this.indices, this.values, 0, this.indices.length - 1);
    }

    /** Sorts the indices and values using quick sort. */
    private static void sortImpl(long[] indices, double[] values, int low, int high) {
        int pivotPos = (low + high) / 2;
        long pivot = indices[pivotPos];
        swapIndexAndValue(indices, values, pivotPos, high);

        int pos = low - 1;
        for (int i = low; i <= high; i++) {
            if (indices[i] <= pivot) {
                pos++;
                swapIndexAndValue(indices, values, pos, i);
            }
        }
        if (high > pos + 1) {
            sortImpl(indices, values, pos + 1, high);
        }
        if (pos - 1 > low) {
            sortImpl(indices, values, low, pos - 1);
        }
    }

    private static void swapIndexAndValue(long[] indices, double[] values, int index1, int index2) {
        long tempIndex = indices[index1];
        indices[index1] = indices[index2];
        indices[index2] = tempIndex;
        double tempValue = values[index1];
        values[index1] = values[index2];
        values[index2] = tempValue;
    }

    @Override
    public SparseLongDoubleVector clone() {
        return new SparseLongDoubleVector(n, indices.clone(), values.clone());
    }
}
