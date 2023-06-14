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
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.Bits;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/** Specialized serializer for {@link DenseIntDoubleVector}. */
public final class DenseIntDoubleVectorSerializer extends TypeSerializer<DenseIntDoubleVector> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY = new double[0];

    private final byte[] buf = new byte[1024];

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<DenseIntDoubleVector> duplicate() {
        return new DenseIntDoubleVectorSerializer();
    }

    @Override
    public DenseIntDoubleVector createInstance() {
        return Vectors.dense(EMPTY);
    }

    @Override
    public DenseIntDoubleVector copy(DenseIntDoubleVector from) {
        return Vectors.dense(Arrays.copyOf(from.getValues(), from.getValues().length));
    }

    @Override
    public DenseIntDoubleVector copy(DenseIntDoubleVector from, DenseIntDoubleVector reuse) {
        if (from.getValues().length == reuse.getValues().length) {
            System.arraycopy(from.getValues(), 0, reuse.getValues(), 0, from.getValues().length);
            return reuse;
        }
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(DenseIntDoubleVector vector, DataOutputView target) throws IOException {
        if (vector == null) {
            throw new IllegalArgumentException("The vector must not be null.");
        }

        double[] values = vector.getValues();
        int len = values.length;
        target.writeInt(len);

        for (int i = 0; i < values.length; i++) {
            Bits.putDouble(buf, (i & 127) << 3, values[i]);
            if ((i & 127) == 127) {
                target.write(buf);
            }
        }
        target.write(buf, 0, (len & 127) << 3);
    }

    @Override
    public DenseIntDoubleVector deserialize(DataInputView source) throws IOException {
        int len = source.readInt();
        double[] values = new double[len];
        readDoubleArray(values, source, len);
        return Vectors.dense(values);
    }

    // Reads `len` double values from `source` into `dst`.
    private void readDoubleArray(double[] dst, DataInputView source, int len) throws IOException {
        int index = 0;
        for (int i = 0; i < (len >> 7); i++) {
            source.readFully(buf, 0, 1024);
            for (int j = 0; j < 128; j++) {
                dst[index++] = Bits.getDouble(buf, j << 3);
            }
        }
        source.readFully(buf, 0, (len << 3) & 1023);
        for (int j = 0; j < (len & 127); j++) {
            dst[index++] = Bits.getDouble(buf, j << 3);
        }
    }

    @Override
    public DenseIntDoubleVector deserialize(DenseIntDoubleVector reuse, DataInputView source)
            throws IOException {
        int len = source.readInt();
        if (len == reuse.getValues().length) {
            readDoubleArray(reuse.getValues(), source, len);
            return reuse;
        }

        double[] values = new double[len];
        readDoubleArray(values, source, len);
        return Vectors.dense(values);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        final int len = source.readInt();
        target.writeInt(len);
        target.write(source, len * 8);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof DenseIntDoubleVectorSerializer;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(DenseIntDoubleVectorSerializer.class);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<DenseIntDoubleVector> snapshotConfiguration() {
        return new DenseVectorSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class DenseVectorSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<DenseIntDoubleVector> {

        public DenseVectorSerializerSnapshot() {
            super(DenseIntDoubleVectorSerializer::new);
        }
    }
}
