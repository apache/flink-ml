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

import org.eclipse.collections.api.tuple.primitive.IntIntPair;
import org.eclipse.collections.impl.tuple.primitive.PrimitiveTuples;

import java.io.IOException;

/** Serializer for {@link IntIntPair}. */
public class IntIntPairSerializer extends TypeSerializerSingleton<IntIntPair> {

    public static final IntIntPairSerializer INSTANCE = new IntIntPairSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public IntIntPair createInstance() {
        return PrimitiveTuples.pair(0, 0);
    }

    @Override
    public IntIntPair copy(IntIntPair from) {
        return PrimitiveTuples.pair(from.getOne(), from.getTwo());
    }

    @Override
    public IntIntPair copy(IntIntPair from, IntIntPair reuse) {
        return copy(from);
    }

    @Override
    public int getLength() {
        return 2 * IntSerializer.INSTANCE.getLength();
    }

    @Override
    public void serialize(IntIntPair record, DataOutputView target) throws IOException {
        IntSerializer.INSTANCE.serialize(record.getOne(), target);
        IntSerializer.INSTANCE.serialize(record.getTwo(), target);
    }

    @Override
    public IntIntPair deserialize(DataInputView source) throws IOException {
        return PrimitiveTuples.pair(
                (int) IntSerializer.INSTANCE.deserialize(source),
                (int) IntSerializer.INSTANCE.deserialize(source));
    }

    @Override
    public IntIntPair deserialize(IntIntPair reuse, DataInputView source) throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<IntIntPair> snapshotConfiguration() {
        return new IntIntPairSerializer.IntIntPairSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class IntIntPairSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<IntIntPair> {

        public IntIntPairSerializerSnapshot() {
            super(IntIntPairSerializer::new);
        }
    }
}
