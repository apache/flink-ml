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
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.LearningNode;

import java.io.IOException;

/** Serializer for {@link LearningNode}. */
public final class LearningNodeSerializer extends TypeSerializerSingleton<LearningNode> {

    public static final LearningNodeSerializer INSTANCE = new LearningNodeSerializer();
    private static final long serialVersionUID = 1L;

    private static final SliceSerializer SLICE_SERIALIZER = SliceSerializer.INSTANCE;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public LearningNode createInstance() {
        return new LearningNode();
    }

    @Override
    public LearningNode copy(LearningNode from) {
        LearningNode learningNode = new LearningNode();
        learningNode.nodeIndex = from.nodeIndex;
        SLICE_SERIALIZER.copy(from.slice, learningNode.slice);
        SLICE_SERIALIZER.copy(from.oob, learningNode.oob);
        learningNode.slice = from.slice;
        return learningNode;
    }

    @Override
    public LearningNode copy(LearningNode from, LearningNode reuse) {
        assert from.getClass() == reuse.getClass();
        reuse.nodeIndex = from.nodeIndex;
        SLICE_SERIALIZER.copy(from.slice, reuse.slice);
        SLICE_SERIALIZER.copy(from.oob, reuse.oob);
        reuse.depth = from.depth;
        return reuse;
    }

    @Override
    public int getLength() {
        return SLICE_SERIALIZER.getLength() + 2 * IntSerializer.INSTANCE.getLength();
    }

    @Override
    public void serialize(LearningNode record, DataOutputView target) throws IOException {
        IntSerializer.INSTANCE.serialize(record.nodeIndex, target);
        SLICE_SERIALIZER.serialize(record.slice, target);
        SLICE_SERIALIZER.serialize(record.oob, target);
        IntSerializer.INSTANCE.serialize(record.depth, target);
    }

    @Override
    public LearningNode deserialize(DataInputView source) throws IOException {
        LearningNode learningNode = new LearningNode();
        learningNode.nodeIndex = IntSerializer.INSTANCE.deserialize(source);
        learningNode.slice = SLICE_SERIALIZER.deserialize(source);
        learningNode.oob = SLICE_SERIALIZER.deserialize(source);
        learningNode.depth = IntSerializer.INSTANCE.deserialize(source);
        return learningNode;
    }

    @Override
    public LearningNode deserialize(LearningNode reuse, DataInputView source) throws IOException {
        reuse.nodeIndex = IntSerializer.INSTANCE.deserialize(source);
        reuse.slice = SLICE_SERIALIZER.deserialize(source);
        reuse.oob = SLICE_SERIALIZER.deserialize(source);
        reuse.depth = IntSerializer.INSTANCE.deserialize(source);
        return reuse;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<LearningNode> snapshotConfiguration() {
        return new LearningNodeSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class LearningNodeSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<LearningNode> {

        public LearningNodeSerializerSnapshot() {
            super(LearningNodeSerializer::new);
        }
    }
}
