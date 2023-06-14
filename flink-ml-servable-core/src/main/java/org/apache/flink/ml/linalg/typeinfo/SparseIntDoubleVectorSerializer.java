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
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;

import java.io.IOException;
import java.util.Arrays;

/** Specialized serializer for {@link SparseIntDoubleVector}. */
public final class SparseIntDoubleVectorSerializer
        extends TypeSerializerSingleton<SparseIntDoubleVector> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY_DOUBLE_ARRAY = new double[0];

    private static final int[] EMPTY_INT_ARRAY = new int[0];

    public static final SparseIntDoubleVectorSerializer INSTANCE =
            new SparseIntDoubleVectorSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public SparseIntDoubleVector createInstance() {
        return Vectors.sparse(0, EMPTY_INT_ARRAY, EMPTY_DOUBLE_ARRAY);
    }

    @Override
    public SparseIntDoubleVector copy(SparseIntDoubleVector from) {
        return Vectors.sparse(
                from.size(),
                Arrays.copyOf(from.getIndices(), from.getIndices().length),
                Arrays.copyOf(from.getValues(), from.getValues().length));
    }

    @Override
    public SparseIntDoubleVector copy(SparseIntDoubleVector from, SparseIntDoubleVector reuse) {
        if (from.getValues().length == reuse.getValues().length && from.size() == reuse.size()) {
            System.arraycopy(from.getValues(), 0, reuse.getValues(), 0, from.getValues().length);
            System.arraycopy(from.getIndices(), 0, reuse.getIndices(), 0, from.getIndices().length);
            return reuse;
        }
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(SparseIntDoubleVector vector, DataOutputView target) throws IOException {
        if (vector == null) {
            throw new IllegalArgumentException("The vector must not be null.");
        }

        target.writeLong(vector.size());
        final int len = vector.getValues().length;
        target.writeInt(len);
        // TODO: optimize the serialization/deserialization process of SparseVectorSerializer.
        int[] indices = vector.getIndices();
        double[] values = vector.getValues();
        for (int i = 0; i < len; i++) {
            target.writeInt(indices[i]);
            target.writeDouble(values[i]);
        }
    }

    // Reads `len` int values from `source` into `indices` and `len` double values from `source`
    // into `values`.
    private void readSparseVectorArrays(
            int[] indices, double[] values, DataInputView source, int len) throws IOException {
        for (int i = 0; i < len; i++) {
            indices[i] = source.readInt();
            values[i] = source.readDouble();
        }
    }

    @Override
    public SparseIntDoubleVector deserialize(DataInputView source) throws IOException {
        long n = source.readLong();
        int len = source.readInt();
        int[] indices = new int[len];
        double[] values = new double[len];
        readSparseVectorArrays(indices, values, source, len);
        return Vectors.sparse(n, indices, values);
    }

    @Override
    public SparseIntDoubleVector deserialize(SparseIntDoubleVector reuse, DataInputView source)
            throws IOException {
        long n = source.readLong();
        int len = source.readInt();
        if (reuse.size() == n && reuse.getValues().length == len) {
            readSparseVectorArrays(reuse.getIndices(), reuse.getValues(), source, len);
            return reuse;
        }

        int[] indices = new int[len];
        double[] values = new double[len];
        readSparseVectorArrays(indices, values, source, len);
        return Vectors.sparse(n, indices, values);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        int n = source.readInt();
        int len = source.readInt();

        target.writeInt(n);
        target.writeInt(len);

        target.write(source, len * 12);
    }

    @Override
    public TypeSerializerSnapshot<SparseIntDoubleVector> snapshotConfiguration() {
        return new SparseVectorSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class SparseVectorSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<SparseIntDoubleVector> {

        public SparseVectorSerializerSnapshot() {
            super(() -> INSTANCE);
        }
    }
}
