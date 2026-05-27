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

package org.apache.flink.ml.linalg.typeinfo;

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.util.Bits;

import java.io.IOException;
import java.util.Objects;

/** A serializer for double arrays. */
@Internal
public final class OptimizedDoublePrimitiveArraySerializer extends TypeSerializer<double[]> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY = new double[0];

    private static final int BUFFER_SIZE = 1024;
    private final byte[] buf = new byte[BUFFER_SIZE];

    public OptimizedDoublePrimitiveArraySerializer() {}

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<double[]> duplicate() {
        return new OptimizedDoublePrimitiveArraySerializer();
    }

    @Override
    public double[] createInstance() {
        return EMPTY;
    }

    @Override
    public double[] copy(double[] from) {
        double[] copy = new double[from.length];
        System.arraycopy(from, 0, copy, 0, from.length);
        return copy;
    }

    @Override
    public double[] copy(double[] from, double[] reuse) {
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(double[] record, DataOutputView target) throws IOException {
        if (record == null) {
            throw new IllegalArgumentException("The record must not be null.");
        }
        serialize(record, 0, record.length, target);
    }

    public void serialize(double[] record, int start, int len, DataOutputView target)
            throws IOException {
        target.writeInt(len);
        for (int i = 0; i < len; i += 1) {
            Bits.putDouble(buf, (i & 127) << 3, record[start + i]);
            if ((i & 127) == 127) {
                target.write(buf);
            }
        }
        target.write(buf, 0, (len & 127) << 3);
    }

    @Override
    public double[] deserialize(DataInputView source) throws IOException {
        final int len = source.readInt();
        double[] result = new double[len];
        readDoubleArray(len, result, source);
        return result;
    }

    public void readDoubleArray(int len, double[] result, DataInputView source) throws IOException {
        int index = 0;
        for (int i = 0; i < (len >> 7); i++) {
            source.readFully(buf, 0, 1024);
            for (int j = 0; j < 128; j++) {
                result[index++] = Bits.getDouble(buf, j << 3);
            }
        }
        source.readFully(buf, 0, (len & 127) << 3);
        for (int j = 0; j < (len & 127); j++) {
            result[index++] = Bits.getDouble(buf, j << 3);
        }
    }

    @Override
    public double[] deserialize(double[] reuse, DataInputView source) throws IOException {
        int len = source.readInt();
        if (len == reuse.length) {
            readDoubleArray(len, reuse, source);
            return reuse;
        }
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        final int len = source.readInt();
        target.writeInt(len);
        target.write(source, len * Double.BYTES);
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof OptimizedDoublePrimitiveArraySerializer;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(OptimizedDoublePrimitiveArraySerializer.class);
    }

    @Override
    public TypeSerializerSnapshot<double[]> snapshotConfiguration() {
        return new DoublePrimitiveArraySerializerSnapshot();
    }

    // ------------------------------------------------------------------------

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class DoublePrimitiveArraySerializerSnapshot
            extends SimpleTypeSerializerSnapshot<double[]> {

        public DoublePrimitiveArraySerializerSnapshot() {
            super(OptimizedDoublePrimitiveArraySerializer::new);
        }
    }
}
