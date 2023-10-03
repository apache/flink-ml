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
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;

import java.io.IOException;

/** Serializer for {@link BinnedInstance}. */
public final class BinnedInstanceSerializer extends TypeSerializerSingleton<BinnedInstance> {

    public static final BinnedInstanceSerializer INSTANCE = new BinnedInstanceSerializer();
    private static final long serialVersionUID = 1L;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public BinnedInstance createInstance() {
        return new BinnedInstance();
    }

    @Override
    public BinnedInstance copy(BinnedInstance from) {
        BinnedInstance instance = new BinnedInstance();
        instance.featureIds = null == from.featureIds ? null : from.featureIds.clone();
        instance.featureValues = from.featureValues.clone();
        instance.label = from.label;
        instance.weight = from.weight;
        return instance;
    }

    @Override
    public BinnedInstance copy(BinnedInstance from, BinnedInstance reuse) {
        assert from.getClass() == reuse.getClass();
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(BinnedInstance record, DataOutputView target) throws IOException {
        if (null == record.featureIds) {
            target.writeBoolean(true);
        } else {
            target.writeBoolean(false);
            IntPrimitiveArraySerializer.INSTANCE.serialize(record.featureIds, target);
        }
        IntPrimitiveArraySerializer.INSTANCE.serialize(record.featureValues, target);
        DoubleSerializer.INSTANCE.serialize(record.label, target);
        DoubleSerializer.INSTANCE.serialize(record.weight, target);
    }

    @Override
    public BinnedInstance deserialize(DataInputView source) throws IOException {
        BinnedInstance instance = new BinnedInstance();
        if (source.readBoolean()) {
            instance.featureIds = null;
        } else {
            instance.featureIds = IntPrimitiveArraySerializer.INSTANCE.deserialize(source);
        }
        instance.featureValues = IntPrimitiveArraySerializer.INSTANCE.deserialize(source);
        instance.label = DoubleSerializer.INSTANCE.deserialize(source);
        instance.weight = DoubleSerializer.INSTANCE.deserialize(source);
        return instance;
    }

    @Override
    public BinnedInstance deserialize(BinnedInstance reuse, DataInputView source)
            throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<BinnedInstance> snapshotConfiguration() {
        return new BinnedInstanceSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class BinnedInstanceSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<BinnedInstance> {

        public BinnedInstanceSerializerSnapshot() {
            super(BinnedInstanceSerializer::new);
        }
    }
}
