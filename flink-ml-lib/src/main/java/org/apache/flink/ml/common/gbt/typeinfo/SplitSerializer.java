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
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.Split;

import java.io.IOException;

/** Specialized serializer for {@link Split}. */
public final class SplitSerializer extends TypeSerializerSingleton<Split> {

    private static final long serialVersionUID = 1L;

    private static final CategoricalSplitSerializer CATEGORICAL_SPLIT_SERIALIZER =
            CategoricalSplitSerializer.INSTANCE;

    private static final ContinuousSplitSerializer CONTINUOUS_SPLIT_SERIALIZER =
            ContinuousSplitSerializer.INSTANCE;

    public static final SplitSerializer INSTANCE = new SplitSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public Split createInstance() {
        return CATEGORICAL_SPLIT_SERIALIZER.createInstance();
    }

    @Override
    public Split copy(Split from) {
        if (from instanceof Split.CategoricalSplit) {
            return CATEGORICAL_SPLIT_SERIALIZER.copy((Split.CategoricalSplit) from);
        } else {
            return CONTINUOUS_SPLIT_SERIALIZER.copy((Split.ContinuousSplit) from);
        }
    }

    @Override
    public Split copy(Split from, Split reuse) {
        assert from.getClass() == reuse.getClass();
        if (from instanceof Split.CategoricalSplit) {
            return CATEGORICAL_SPLIT_SERIALIZER.copy(
                    (Split.CategoricalSplit) from, (Split.CategoricalSplit) reuse);
        } else {
            return CONTINUOUS_SPLIT_SERIALIZER.copy(
                    (Split.ContinuousSplit) from, (Split.ContinuousSplit) reuse);
        }
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Split record, DataOutputView target) throws IOException {
        if (null == record) {
            target.writeByte(0);
        } else if (record instanceof Split.CategoricalSplit) {
            target.writeByte(1);
            CATEGORICAL_SPLIT_SERIALIZER.serialize((Split.CategoricalSplit) record, target);
        } else {
            target.writeByte(2);
            CONTINUOUS_SPLIT_SERIALIZER.serialize((Split.ContinuousSplit) record, target);
        }
    }

    @Override
    public Split deserialize(DataInputView source) throws IOException {
        byte type = source.readByte();
        if (type == 0) {
            return null;
        } else if (type == 1) {
            return CATEGORICAL_SPLIT_SERIALIZER.deserialize(source);
        } else {
            return CONTINUOUS_SPLIT_SERIALIZER.deserialize(source);
        }
    }

    @Override
    public Split deserialize(Split reuse, DataInputView source) throws IOException {
        byte type = source.readByte();
        if (type == 0) {
            return null;
        }
        assert type == 1 && reuse instanceof Split.CategoricalSplit
                || type == 2 && reuse instanceof Split.ContinuousSplit;
        if (type == 1) {
            return CATEGORICAL_SPLIT_SERIALIZER.deserialize((Split.CategoricalSplit) reuse, source);
        } else {
            return CONTINUOUS_SPLIT_SERIALIZER.deserialize((Split.ContinuousSplit) reuse, source);
        }
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<Split> snapshotConfiguration() {
        return new SplitSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class SplitSerializerSnapshot extends SimpleTypeSerializerSnapshot<Split> {

        public SplitSerializerSnapshot() {
            super(SplitSerializer::new);
        }
    }
}
