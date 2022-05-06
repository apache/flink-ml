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
import org.apache.flink.ml.linalg.SparseVector;

import java.io.IOException;
import java.util.Arrays;

/** Specialized serializer for {@link SparseVector}. */
public final class SparseVectorSerializer extends TypeSerializerSingleton<SparseVector> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY_DOUBLE_ARRAY = new double[0];

    private static final int[] EMPTY_INT_ARRAY = new int[0];

    public static final SparseVectorSerializer INSTANCE = new SparseVectorSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public SparseVector createInstance() {
        return new SparseVector(0, EMPTY_INT_ARRAY, EMPTY_DOUBLE_ARRAY);
    }

    @Override
    public SparseVector copy(SparseVector from) {
        return new SparseVector(
                from.n,
                Arrays.copyOf(from.indices, from.indices.length),
                Arrays.copyOf(from.values, from.values.length));
    }

    @Override
    public SparseVector copy(SparseVector from, SparseVector reuse) {
        if (from.values.length == reuse.values.length && from.n == reuse.n) {
            System.arraycopy(from.values, 0, reuse.values, 0, from.values.length);
            System.arraycopy(from.indices, 0, reuse.indices, 0, from.indices.length);
            return reuse;
        }
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(SparseVector vector, DataOutputView target) throws IOException {
        if (vector == null) {
            throw new IllegalArgumentException("The vector must not be null.");
        }

        target.writeInt(vector.n);
        final int len = vector.values.length;
        target.writeInt(len);
        // TODO: optimize the serialization/deserialization process of SparseVectorSerializer.
        for (int i = 0; i < len; i++) {
            target.writeInt(vector.indices[i]);
            target.writeDouble(vector.values[i]);
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
    public SparseVector deserialize(DataInputView source) throws IOException {
        int n = source.readInt();
        int len = source.readInt();
        int[] indices = new int[len];
        double[] values = new double[len];
        readSparseVectorArrays(indices, values, source, len);
        return new SparseVector(n, indices, values);
    }

    @Override
    public SparseVector deserialize(SparseVector reuse, DataInputView source) throws IOException {
        int n = source.readInt();
        int len = source.readInt();
        if (reuse.n == n && reuse.values.length == len) {
            readSparseVectorArrays(reuse.indices, reuse.values, source, len);
            return reuse;
        }

        int[] indices = new int[len];
        double[] values = new double[len];
        readSparseVectorArrays(indices, values, source, len);
        return new SparseVector(n, indices, values);
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
    public TypeSerializerSnapshot<SparseVector> snapshotConfiguration() {
        return new SparseVectorSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class SparseVectorSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<SparseVector> {

        public SparseVectorSerializerSnapshot() {
            super(() -> INSTANCE);
        }
    }
}
