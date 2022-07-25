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
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfoFactory;

import java.util.Arrays;

/** A dense vector of double values. */
@TypeInfo(DenseVectorTypeInfoFactory.class)
public class DenseVector implements Vector {
    public final double[] values;

    public DenseVector(double[] values) {
        this.values = values;
    }

    public DenseVector(int size) {
        this.values = new double[size];
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public double get(int i) {
        return values[i];
    }

    @Override
    public void set(int i, double value) {
        values[i] = value;
    }

    @Override
    public double[] toArray() {
        return values;
    }

    @Override
    public DenseVector toDense() {
        return this;
    }

    @Override
    public SparseVector toSparse() {
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

        return new SparseVector(size(), nonZeroIndices, numZeroValues);
    }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof DenseVector)) {
            return false;
        }
        return Arrays.equals(values, ((DenseVector) obj).values);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(values);
    }

    @Override
    public DenseVector clone() {
        return new DenseVector(values.clone());
    }
}
