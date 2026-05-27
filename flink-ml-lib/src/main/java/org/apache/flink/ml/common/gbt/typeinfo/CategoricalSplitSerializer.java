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
import org.apache.flink.api.common.typeutils.base.BooleanSerializer;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.api.common.typeutils.base.array.BytePrimitiveArraySerializer;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.Split;

import java.io.IOException;
import java.util.BitSet;

/** Specialized serializer for {@link Split.CategoricalSplit}. */
public final class CategoricalSplitSerializer
        extends TypeSerializerSingleton<Split.CategoricalSplit> {

    private static final long serialVersionUID = 1L;

    public static final CategoricalSplitSerializer INSTANCE = new CategoricalSplitSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public Split.CategoricalSplit createInstance() {
        return new Split.CategoricalSplit(-1, Split.INVALID_GAIN, 0, false, 0., new BitSet());
    }

    @Override
    public Split.CategoricalSplit copy(Split.CategoricalSplit from) {
        return new Split.CategoricalSplit(
                from.featureId,
                from.gain,
                from.missingBin,
                from.missingGoLeft,
                from.prediction,
                from.categoriesGoLeft);
    }

    @Override
    public Split.CategoricalSplit copy(Split.CategoricalSplit from, Split.CategoricalSplit reuse) {
        assert from.getClass() == reuse.getClass();
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Split.CategoricalSplit record, DataOutputView target) throws IOException {
        IntSerializer.INSTANCE.serialize(record.featureId, target);
        DoubleSerializer.INSTANCE.serialize(record.gain, target);
        IntSerializer.INSTANCE.serialize(record.missingBin, target);
        BooleanSerializer.INSTANCE.serialize(record.missingGoLeft, target);
        DoubleSerializer.INSTANCE.serialize(record.prediction, target);
        BytePrimitiveArraySerializer.INSTANCE.serialize(
                record.categoriesGoLeft.toByteArray(), target);
    }

    @Override
    public Split.CategoricalSplit deserialize(DataInputView source) throws IOException {
        return new Split.CategoricalSplit(
                IntSerializer.INSTANCE.deserialize(source),
                DoubleSerializer.INSTANCE.deserialize(source),
                IntSerializer.INSTANCE.deserialize(source),
                BooleanSerializer.INSTANCE.deserialize(source),
                DoubleSerializer.INSTANCE.deserialize(source),
                BitSet.valueOf(BytePrimitiveArraySerializer.INSTANCE.deserialize(source)));
    }

    @Override
    public Split.CategoricalSplit deserialize(Split.CategoricalSplit reuse, DataInputView source)
            throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<Split.CategoricalSplit> snapshotConfiguration() {
        return new CategoricalSplitSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class CategoricalSplitSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<Split.CategoricalSplit> {

        public CategoricalSplitSerializerSnapshot() {
            super(CategoricalSplitSerializer::new);
        }
    }
}
