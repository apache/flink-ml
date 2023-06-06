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
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.VectorWithNorm;

import java.io.IOException;

/** Specialized serializer for {@link VectorWithNorm}. */
public class VectorWithNormSerializer extends TypeSerializer<VectorWithNorm> {
    private final VectorSerializer vectorSerializer = new VectorSerializer();

    private static final long serialVersionUID = 1L;

    private static final double[] EMPTY = new double[0];

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<VectorWithNorm> duplicate() {
        return new VectorWithNormSerializer();
    }

    @Override
    public VectorWithNorm createInstance() {
        return new VectorWithNorm(new DenseIntDoubleVector(EMPTY));
    }

    @Override
    public VectorWithNorm copy(VectorWithNorm from) {
        IntDoubleVector vector = (IntDoubleVector) vectorSerializer.copy(from.vector);
        return new VectorWithNorm(vector, from.l2Norm);
    }

    @Override
    public VectorWithNorm copy(VectorWithNorm from, VectorWithNorm reuse) {
        IntDoubleVector vector = (IntDoubleVector) vectorSerializer.copy(from.vector, reuse.vector);
        return new VectorWithNorm(vector, from.l2Norm);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(VectorWithNorm from, DataOutputView dataOutputView) throws IOException {
        vectorSerializer.serialize(from.vector, dataOutputView);
        dataOutputView.writeDouble(from.l2Norm);
    }

    @Override
    public VectorWithNorm deserialize(DataInputView dataInputView) throws IOException {
        IntDoubleVector vector = (IntDoubleVector) vectorSerializer.deserialize(dataInputView);
        double l2NormSquare = dataInputView.readDouble();
        return new VectorWithNorm(vector, l2NormSquare);
    }

    @Override
    public VectorWithNorm deserialize(VectorWithNorm reuse, DataInputView dataInputView)
            throws IOException {
        IntDoubleVector vector =
                (IntDoubleVector) vectorSerializer.deserialize(reuse.vector, dataInputView);
        double l2NormSquare = dataInputView.readDouble();
        return new VectorWithNorm(vector, l2NormSquare);
    }

    @Override
    public void copy(DataInputView dataInputView, DataOutputView dataOutputView)
            throws IOException {
        vectorSerializer.copy(dataInputView, dataOutputView);
        dataOutputView.write(dataInputView, 8);
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof VectorWithNormSerializer;
    }

    @Override
    public int hashCode() {
        return VectorWithNormSerializer.class.hashCode();
    }

    @Override
    public TypeSerializerSnapshot<VectorWithNorm> snapshotConfiguration() {
        return new VectorWithNormSerializerSnapshot();
    }

    private static class VectorWithNormSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<VectorWithNorm> {
        public VectorWithNormSerializerSnapshot() {
            super(VectorWithNormSerializer::new);
        }
    }
}
