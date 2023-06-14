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
import org.apache.flink.ml.linalg.SparseLongDoubleVector;
import org.apache.flink.ml.linalg.Vectors;

import java.io.IOException;
import java.util.Arrays;

/** Specialized serializer for {@link SparseLongDoubleVector}. */
public final class SparseLongDoubleVectorSerializer
        extends TypeSerializerSingleton<SparseLongDoubleVector> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY_DOUBLE_ARRAY = new double[0];

    private static final long[] EMPTY_LONG_ARRAY = new long[0];

    public static final SparseLongDoubleVectorSerializer INSTANCE =
            new SparseLongDoubleVectorSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public SparseLongDoubleVector createInstance() {
        return Vectors.sparse(0, EMPTY_LONG_ARRAY, EMPTY_DOUBLE_ARRAY);
    }

    @Override
    public SparseLongDoubleVector copy(SparseLongDoubleVector from) {
        long[] indices = from.getIndices();
        double[] values = from.getValues();
        return Vectors.sparse(
                from.size(),
                Arrays.copyOf(indices, indices.length),
                Arrays.copyOf(values, values.length));
    }

    @Override
    public SparseLongDoubleVector copy(SparseLongDoubleVector from, SparseLongDoubleVector reuse) {
        long[] fromIndices = from.getIndices();
        double[] fromValues = from.getValues();
        long[] reuseIndices = reuse.getIndices();
        double[] reuseValues = reuse.getValues();
        if (fromValues.length == reuseValues.length && from.size() == reuse.size()) {
            System.arraycopy(fromValues, 0, reuseValues, 0, fromValues.length);
            System.arraycopy(fromIndices, 0, reuseIndices, 0, fromIndices.length);
            return reuse;
        }
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(SparseLongDoubleVector vector, DataOutputView target) throws IOException {
        if (vector == null) {
            throw new IllegalArgumentException("The vector must not be null.");
        }

        target.writeLong(vector.size());
        final int len = vector.getValues().length;
        target.writeInt(len);
        // TODO: optimize the serialization/deserialization process of SparseVectorSerializer.
        long[] indices = vector.getIndices();
        double[] values = vector.getValues();
        for (int i = 0; i < len; i++) {
            target.writeLong(indices[i]);
            target.writeDouble(values[i]);
        }
    }

    // Reads `len` int values from `source` into `indices` and `len` double values from `source`
    // into `values`.
    private void readSparseVectorArrays(
            long[] indices, double[] values, DataInputView source, int len) throws IOException {
        for (int i = 0; i < len; i++) {
            indices[i] = source.readLong();
            values[i] = source.readDouble();
        }
    }

    @Override
    public SparseLongDoubleVector deserialize(DataInputView source) throws IOException {
        long n = source.readLong();
        int len = source.readInt();
        long[] indices = new long[len];
        double[] values = new double[len];
        readSparseVectorArrays(indices, values, source, len);
        return Vectors.sparse(n, indices, values);
    }

    @Override
    public SparseLongDoubleVector deserialize(SparseLongDoubleVector reuse, DataInputView source)
            throws IOException {
        long n = source.readLong();
        int len = source.readInt();
        long[] indices = reuse.getIndices();
        double[] values = reuse.getValues();
        if (reuse.size() == n && values.length == len) {
            readSparseVectorArrays(indices, values, source, len);
            return reuse;
        }

        indices = new long[len];
        values = new double[len];
        readSparseVectorArrays(indices, values, source, len);
        return Vectors.sparse(n, indices, values);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        long n = source.readLong();
        int len = source.readInt();

        target.writeLong(n);
        target.writeInt(len);

        target.write(source, len * 16);
    }

    @Override
    public TypeSerializerSnapshot<SparseLongDoubleVector> snapshotConfiguration() {
        return new SparseLongDoubleVectorSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class SparseLongDoubleVectorSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<SparseLongDoubleVector> {

        public SparseLongDoubleVectorSerializerSnapshot() {
            super(() -> INSTANCE);
        }
    }
}
