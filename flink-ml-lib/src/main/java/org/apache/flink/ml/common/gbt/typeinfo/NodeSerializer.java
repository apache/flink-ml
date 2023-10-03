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
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.Node;

import java.io.IOException;

/** Serializer for {@link Node}. */
public final class NodeSerializer extends TypeSerializerSingleton<Node> {

    public static final NodeSerializer INSTANCE = new NodeSerializer();
    private static final long serialVersionUID = 1L;

    private static final SplitSerializer SPLIT_SERIALIZER = SplitSerializer.INSTANCE;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public Node createInstance() {
        return new Node();
    }

    @Override
    public Node copy(Node from) {
        Node node = new Node();
        node.split = SPLIT_SERIALIZER.copy(from.split);
        node.isLeaf = from.isLeaf;
        node.left = from.left;
        node.right = from.right;
        return node;
    }

    @Override
    public Node copy(Node from, Node reuse) {
        assert from.getClass() == reuse.getClass();
        SPLIT_SERIALIZER.copy(from.split, reuse.split);
        reuse.isLeaf = from.isLeaf;
        reuse.left = from.left;
        reuse.right = from.right;
        return reuse;
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Node record, DataOutputView target) throws IOException {
        SPLIT_SERIALIZER.serialize(record.split, target);
        BooleanSerializer.INSTANCE.serialize(record.isLeaf, target);
        IntSerializer.INSTANCE.serialize(record.left, target);
        IntSerializer.INSTANCE.serialize(record.right, target);
    }

    @Override
    public Node deserialize(DataInputView source) throws IOException {
        Node node = new Node();
        node.split = SPLIT_SERIALIZER.deserialize(source);
        node.isLeaf = BooleanSerializer.INSTANCE.deserialize(source);
        node.left = IntSerializer.INSTANCE.deserialize(source);
        node.right = IntSerializer.INSTANCE.deserialize(source);
        return node;
    }

    @Override
    public Node deserialize(Node reuse, DataInputView source) throws IOException {
        reuse.split = SPLIT_SERIALIZER.deserialize(source);
        reuse.isLeaf = BooleanSerializer.INSTANCE.deserialize(source);
        reuse.left = IntSerializer.INSTANCE.deserialize(source);
        reuse.right = IntSerializer.INSTANCE.deserialize(source);
        return reuse;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<Node> snapshotConfiguration() {
        return new NodeSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class NodeSerializerSnapshot extends SimpleTypeSerializerSnapshot<Node> {

        public NodeSerializerSnapshot() {
            super(NodeSerializer::new);
        }
    }
}
