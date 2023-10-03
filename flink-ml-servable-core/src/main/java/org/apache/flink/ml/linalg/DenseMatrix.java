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
import org.apache.flink.ml.linalg.typeinfo.DenseMatrixTypeInfoFactory;
import org.apache.flink.util.Preconditions;

/**
 * Column-major dense matrix. The entry values are stored in a single array of doubles with columns
 * listed in sequence.
 */
@TypeInfo(DenseMatrixTypeInfoFactory.class)
@PublicEvolving
public class DenseMatrix implements Matrix {

    /** Row dimension. */
    private final int numRows;

    /** Column dimension. */
    private final int numCols;

    /**
     * Array for internal storage of elements.
     *
     * <p>The matrix data is stored in column major format internally.
     */
    public final double[] values;

    /**
     * Constructs an m-by-n matrix of zeros.
     *
     * @param numRows Number of rows.
     * @param numCols Number of columns.
     */
    public DenseMatrix(int numRows, int numCols) {
        this(numRows, numCols, new double[numRows * numCols]);
    }

    /**
     * Constructs a matrix from a 1-D array. The data in the array should be organized in column
     * major.
     *
     * @param numRows Number of rows.
     * @param numCols Number of cols.
     * @param values One-dimensional array of doubles.
     */
    public DenseMatrix(int numRows, int numCols, double[] values) {
        Preconditions.checkArgument(values.length == numRows * numCols);
        this.numRows = numRows;
        this.numCols = numCols;
        this.values = values;
    }

    @Override
    public int numRows() {
        return numRows;
    }

    @Override
    public int numCols() {
        return numCols;
    }

    @Override
    public double get(int i, int j) {
        Preconditions.checkArgument(i >= 0 && i < numRows && j >= 0 && j < numCols);
        return values[numRows * j + i];
    }

    @Override
    public double add(int i, int j, double value) {
        Preconditions.checkArgument(i >= 0 && i < numRows && j >= 0 && j < numCols);
        return values[numRows * j + i] += value;
    }

    @Override
    public double set(int i, int j, double value) {
        Preconditions.checkArgument(i >= 0 && i < numRows && j >= 0 && j < numCols);
        return values[numRows * j + i] = value;
    }

    @Override
    public DenseMatrix toDense() {
        return this;
    }
}
