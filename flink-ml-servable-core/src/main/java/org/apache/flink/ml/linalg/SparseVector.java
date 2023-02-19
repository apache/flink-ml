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

package org.apache.flink.ml.linalg;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfoFactory;
import org.apache.flink.util.Preconditions;

import java.util.Arrays;
import java.util.Objects;

/** A sparse vector of double values. */
@TypeInfo(SparseVectorTypeInfoFactory.class)
@PublicEvolving
public class SparseVector implements Vector {
    public final int n;
    public int[] indices;
    public double[] values;

    public SparseVector(int n, int[] indices, double[] values) {
        this.n = n;
        this.indices = indices;
        this.values = values;
        if (!isIndicesSorted()) {
            sortIndices();
        }
        validateSortedData();
    }

    @Override
    public int size() {
        return n;
    }

    @Override
    public double get(int i) {
        int pos = Arrays.binarySearch(indices, i);
        if (pos >= 0) {
            return values[pos];
        }
        return 0.;
    }

    @Override
    public void set(int i, double value) {
        int pos = Arrays.binarySearch(indices, i);
        if (pos >= 0) {
            values[pos] = value;
        } else if (value != 0.0) {
            Preconditions.checkArgument(i < n, "Index out of bounds: " + i);
            int[] indices = new int[this.indices.length + 1];
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
    public double[] toArray() {
        double[] result = new double[n];
        for (int i = 0; i < indices.length; i++) {
            result[indices[i]] = values[i];
        }
        return result;
    }

    @Override
    public DenseVector toDense() {
        return new DenseVector(toArray());
    }

    @Override
    public SparseVector toSparse() {
        return this;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        SparseVector that = (SparseVector) o;
        return n == that.n
                && Arrays.equals(indices, that.indices)
                && Arrays.equals(values, that.values);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(n);
        result = 31 * result + Arrays.hashCode(indices);
        result = 31 * result + Arrays.hashCode(values);
        return result;
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
    private static void sortImpl(int[] indices, double[] values, int low, int high) {
        int pivotPos = (low + high) / 2;
        int pivot = indices[pivotPos];
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

    private static void swapIndexAndValue(int[] indices, double[] values, int index1, int index2) {
        int tempIndex = indices[index1];
        indices[index1] = indices[index2];
        indices[index2] = tempIndex;
        double tempValue = values[index1];
        values[index1] = values[index2];
        values[index2] = tempValue;
    }

    @Override
    public String toString() {
        String sbr =
                "(" + n + ", " + Arrays.toString(indices) + ", " + Arrays.toString(values) + ")";
        return sbr;
    }

    @Override
    public SparseVector clone() {
        return new SparseVector(n, indices.clone(), values.clone());
    }
}
