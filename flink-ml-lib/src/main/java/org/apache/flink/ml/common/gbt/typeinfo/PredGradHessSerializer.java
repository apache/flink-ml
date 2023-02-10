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

package org.apache.flink.ml.common.gbt.typeinfo;

import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;

import java.io.IOException;

/** Serializer for {@link PredGradHess}. */
public final class PredGradHessSerializer extends TypeSerializerSingleton<PredGradHess> {

    public static final PredGradHessSerializer INSTANCE = new PredGradHessSerializer();
    private static final long serialVersionUID = 1L;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public PredGradHess createInstance() {
        return new PredGradHess();
    }

    @Override
    public PredGradHess copy(PredGradHess from) {
        PredGradHess instance = new PredGradHess();
        instance.pred = from.pred;
        instance.gradient = from.gradient;
        instance.hessian = from.hessian;
        return instance;
    }

    @Override
    public PredGradHess copy(PredGradHess from, PredGradHess reuse) {
        assert from.getClass() == reuse.getClass();
        reuse.pred = from.pred;
        reuse.gradient = from.gradient;
        reuse.hessian = from.hessian;
        return reuse;
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(PredGradHess record, DataOutputView target) throws IOException {
        DoubleSerializer.INSTANCE.serialize(record.pred, target);
        DoubleSerializer.INSTANCE.serialize(record.gradient, target);
        DoubleSerializer.INSTANCE.serialize(record.hessian, target);
    }

    @Override
    public PredGradHess deserialize(DataInputView source) throws IOException {
        PredGradHess instance = new PredGradHess();
        instance.pred = DoubleSerializer.INSTANCE.deserialize(source);
        instance.gradient = DoubleSerializer.INSTANCE.deserialize(source);
        instance.hessian = DoubleSerializer.INSTANCE.deserialize(source);
        return instance;
    }

    @Override
    public PredGradHess deserialize(PredGradHess reuse, DataInputView source) throws IOException {
        reuse.pred = DoubleSerializer.INSTANCE.deserialize(source);
        reuse.gradient = DoubleSerializer.INSTANCE.deserialize(source);
        reuse.hessian = DoubleSerializer.INSTANCE.deserialize(source);
        return reuse;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<PredGradHess> snapshotConfiguration() {
        return new PredGradHessSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class PredGradHessSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<PredGradHess> {

        public PredGradHessSerializerSnapshot() {
            super(PredGradHessSerializer::new);
        }
    }
}
