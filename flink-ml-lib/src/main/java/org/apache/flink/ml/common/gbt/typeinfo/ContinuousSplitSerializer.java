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
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.Split;

import java.io.IOException;

/** Specialized serializer for {@link Split.ContinuousSplit}. */
public final class ContinuousSplitSerializer
        extends TypeSerializerSingleton<Split.ContinuousSplit> {

    private static final long serialVersionUID = 1L;

    public static final ContinuousSplitSerializer INSTANCE = new ContinuousSplitSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public Split.ContinuousSplit createInstance() {
        return new Split.ContinuousSplit(-1, Split.INVALID_GAIN, 0, false, 0., 0., false, 0);
    }

    @Override
    public Split.ContinuousSplit copy(Split.ContinuousSplit from) {
        return new Split.ContinuousSplit(
                from.featureId,
                from.gain,
                from.missingBin,
                from.missingGoLeft,
                from.prediction,
                from.threshold,
                from.isUnseenMissing,
                from.zeroBin);
    }

    @Override
    public Split.ContinuousSplit copy(Split.ContinuousSplit from, Split.ContinuousSplit reuse) {
        assert from.getClass() == reuse.getClass();
        return copy(from);
    }

    @Override
    public int getLength() {
        return 3 * IntSerializer.INSTANCE.getLength()
                + 3 * DoubleSerializer.INSTANCE.getLength()
                + 2 * BooleanSerializer.INSTANCE.getLength();
    }

    @Override
    public void serialize(Split.ContinuousSplit record, DataOutputView target) throws IOException {
        target.writeInt(record.featureId);
        target.writeDouble(record.gain);
        target.writeInt(record.missingBin);
        target.writeBoolean(record.missingGoLeft);
        target.writeDouble(record.prediction);
        target.writeDouble(record.threshold);
        target.writeBoolean(record.isUnseenMissing);
        target.writeInt(record.zeroBin);
    }

    @Override
    public Split.ContinuousSplit deserialize(DataInputView source) throws IOException {
        return new Split.ContinuousSplit(
                source.readInt(),
                source.readDouble(),
                source.readInt(),
                source.readBoolean(),
                source.readDouble(),
                source.readDouble(),
                source.readBoolean(),
                source.readInt());
    }

    @Override
    public Split.ContinuousSplit deserialize(Split.ContinuousSplit reuse, DataInputView source)
            throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<Split.ContinuousSplit> snapshotConfiguration() {
        return new ContinuousSplitSplitSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class ContinuousSplitSplitSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<Split.ContinuousSplit> {

        public ContinuousSplitSplitSerializerSnapshot() {
            super(ContinuousSplitSerializer::new);
        }
    }
}
