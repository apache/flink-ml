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
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;

import java.io.IOException;

/** Specialized serializer for {@link Vector}. */
public final class VectorSerializer extends TypeSerializerSingleton<Vector> {

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY = new double[0];

    private final DenseVectorSerializer denseVectorSerializer = new DenseVectorSerializer();

    private static final SparseVectorSerializer SPARSE_VECTOR_SERIALIZER =
            SparseVectorSerializer.INSTANCE;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public Vector createInstance() {
        return new DenseVector(EMPTY);
    }

    @Override
    public Vector copy(Vector from) {
        if (from instanceof DenseVector) {
            return denseVectorSerializer.copy((DenseVector) from);
        } else {
            return SPARSE_VECTOR_SERIALIZER.copy((SparseVector) from);
        }
    }

    @Override
    public Vector copy(Vector from, Vector reuse) {
        assert from.getClass() == reuse.getClass();
        if (from instanceof DenseVector) {
            return denseVectorSerializer.copy((DenseVector) from, (DenseVector) reuse);
        } else {
            return SPARSE_VECTOR_SERIALIZER.copy((SparseVector) from, (SparseVector) reuse);
        }
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Vector vector, DataOutputView target) throws IOException {
        if (vector instanceof DenseVector) {
            target.writeByte(0);
            denseVectorSerializer.serialize((DenseVector) vector, target);
        } else {
            target.writeByte(1);
            SPARSE_VECTOR_SERIALIZER.serialize((SparseVector) vector, target);
        }
    }

    @Override
    public Vector deserialize(DataInputView source) throws IOException {
        byte type = source.readByte();
        if (type == 0) {
            return denseVectorSerializer.deserialize(source);
        } else {
            return SPARSE_VECTOR_SERIALIZER.deserialize(source);
        }
    }

    @Override
    public Vector deserialize(Vector reuse, DataInputView source) throws IOException {
        byte type = source.readByte();
        assert type == 0 && reuse instanceof DenseVector
                || type == 1 && reuse instanceof SparseVector;
        if (type == 0) {
            return denseVectorSerializer.deserialize(source);
        } else {
            return SPARSE_VECTOR_SERIALIZER.deserialize(source);
        }
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<Vector> snapshotConfiguration() {
        return new VectorSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class VectorSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<Vector> {

        public VectorSerializerSnapshot() {
            super(VectorSerializer::new);
        }
    }
}
