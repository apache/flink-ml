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

package org.apache.flink.ml.linalg.typeinfo;

import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.linalg.DenseMatrix;

import java.io.IOException;
import java.util.Arrays;

/** Specialized serializer for {@link DenseMatrix}. */
public final class DenseMatrixSerializer extends TypeSerializerSingleton<DenseMatrix> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY = new double[0];

    public static final DenseMatrixSerializer INSTANCE = new DenseMatrixSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public DenseMatrix createInstance() {
        return new DenseMatrix(0, 0, EMPTY);
    }

    @Override
    public DenseMatrix copy(DenseMatrix from) {
        return new DenseMatrix(
                from.numRows(), from.numCols(), Arrays.copyOf(from.values, from.values.length));
    }

    @Override
    public DenseMatrix copy(DenseMatrix from, DenseMatrix reuse) {
        if (from.values.length == reuse.values.length) {
            System.arraycopy(from.values, 0, reuse.values, 0, from.values.length);
            if (from.numCols() == reuse.numCols()) {
                return reuse;
            } else {
                return new DenseMatrix(from.numRows(), from.numCols(), reuse.values);
            }
        }
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(DenseMatrix matrix, DataOutputView target) throws IOException {
        if (matrix == null) {
            throw new IllegalArgumentException("The matrix must not be null.");
        }
        final int len = matrix.values.length;
        target.writeInt(matrix.numRows());
        target.writeInt(matrix.numCols());
        for (int i = 0; i < len; i++) {
            target.writeDouble(matrix.values[i]);
        }
    }

    @Override
    public DenseMatrix deserialize(DataInputView source) throws IOException {
        int m = source.readInt();
        int n = source.readInt();
        double[] values = new double[m * n];
        deserializeDoubleArray(values, source, m * n);
        return new DenseMatrix(m, n, values);
    }

    private static void deserializeDoubleArray(double[] dst, DataInputView source, int len)
            throws IOException {
        for (int i = 0; i < len; i++) {
            dst[i] = source.readDouble();
        }
    }

    @Override
    public DenseMatrix deserialize(DenseMatrix reuse, DataInputView source) throws IOException {
        int m = source.readInt();
        int n = source.readInt();
        double[] values = reuse.values;
        if (values.length != m * n) {
            double[] tmpValues = new double[m * n];
            deserializeDoubleArray(tmpValues, source, m * n);
            return new DenseMatrix(m, n, tmpValues);
        }
        deserializeDoubleArray(values, source, m * n);
        return new DenseMatrix(m, n, values);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        int m = source.readInt();
        target.writeInt(m);
        int n = source.readInt();
        target.writeInt(n);

        target.write(source, m * n * Double.BYTES);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<DenseMatrix> snapshotConfiguration() {
        return new DenseMatrixSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class DenseMatrixSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<DenseMatrix> {
        public DenseMatrixSerializerSnapshot() {
            super(() -> INSTANCE);
        }
    }
}
