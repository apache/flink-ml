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
import org.apache.flink.ml.linalg.DenseVector;

import java.io.IOException;
import java.util.Arrays;

/** Specialized serializer for {@code DenseVector}. */
public final class DenseVectorSerializer extends TypeSerializerSingleton<DenseVector> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY = new double[0];

    private static final DenseVectorSerializer INSTANCE = new DenseVectorSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public DenseVector createInstance() {
        return new DenseVector(EMPTY);
    }

    @Override
    public DenseVector copy(DenseVector from) {
        return new DenseVector(Arrays.copyOf(from.values, from.values.length));
    }

    @Override
    public DenseVector copy(DenseVector from, DenseVector reuse) {
        if (from.values.length == reuse.values.length) {
            System.arraycopy(from.values, 0, reuse.values, 0, from.values.length);
            return reuse;
        }
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(DenseVector vector, DataOutputView target) throws IOException {
        if (vector == null) {
            throw new IllegalArgumentException("The vector must not be null.");
        }

        final int len = vector.values.length;
        target.writeInt(len);
        for (int i = 0; i < len; i++) {
            target.writeDouble(vector.get(i));
        }
    }

    @Override
    public DenseVector deserialize(DataInputView source) throws IOException {
        int len = source.readInt();
        double[] values = new double[len];
        for (int i = 0; i < len; i++) {
            values[i] = source.readDouble();
        }
        return new DenseVector(values);
    }

    // Reads `len` double values from `source` into `dst`.
    private static void readDoubleArray(double[] dst, DataInputView source, int len)
            throws IOException {
        for (int i = 0; i < len; i++) {
            dst[i] = source.readDouble();
        }
    }

    @Override
    public DenseVector deserialize(DenseVector reuse, DataInputView source) throws IOException {
        int len = source.readInt();
        if (len == reuse.values.length) {
            readDoubleArray(reuse.values, source, len);
            return reuse;
        }

        double[] values = new double[len];
        readDoubleArray(values, source, len);
        return new DenseVector(values);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        final int len = source.readInt();
        target.writeInt(len);
        target.write(source, len * 8);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<DenseVector> snapshotConfiguration() {
        return new DenseVectorSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class DenseVectorSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<DenseVector> {

        public DenseVectorSerializerSnapshot() {
            super(() -> INSTANCE);
        }
    }
}
