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

import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfoFactory;
import org.apache.flink.util.Preconditions;

import java.util.Arrays;

/** A dense vector of int indices and double values. */
@TypeInfo(DenseIntDoubleVectorTypeInfoFactory.class)
public class DenseIntDoubleVector implements DenseVector<Integer, Double, int[], double[]> {
    private final double[] values;

    DenseIntDoubleVector(double[] values) {
        this.values = values;
    }

    DenseIntDoubleVector(long size) {
        Preconditions.checkArgument(
                size < Integer.MAX_VALUE, "Size of dense vector exceeds INT.MAX.");
        this.values = new double[(int) size];
    }

    @Override
    public long size() {
        return values.length;
    }

    @Override
    public Double get(Integer index) {
        return values[index];
    }

    /** Avoids auto-boxing for better performance. */
    public double get(int i) {
        return values[i];
    }

    @Override
    public void set(Integer index, Double value) {
        values[index] = value;
    }

    /** Avoids auto-boxing for better performance. */
    public void set(int i, double value) {
        values[i] = value;
    }

    @Override
    public double[] toArray() {
        return values;
    }

    @Override
    public DenseIntDoubleVector toDense() {
        return this;
    }

    @Override
    public SparseIntDoubleVector toSparse() {
        int numNonZeros = 0;
        for (double value : values) {
            if (value != 0.0) {
                numNonZeros++;
            }
        }

        int[] nonZeroIndices = new int[numNonZeros];
        double[] numZeroValues = new double[numNonZeros];
        int k = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] == 0.0) {
                continue;
            }
            nonZeroIndices[k] = i;
            numZeroValues[k] = values[i];
            k++;
        }

        return new SparseIntDoubleVector(size(), nonZeroIndices, numZeroValues);
    }

    @Override
    public double[] getValues() {
        return values;
    }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof DenseIntDoubleVector)) {
            return false;
        }
        return Arrays.equals(values, ((DenseIntDoubleVector) obj).values);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(values);
    }

    @Override
    public DenseIntDoubleVector clone() {
        return Vectors.dense(values.clone());
    }
}
