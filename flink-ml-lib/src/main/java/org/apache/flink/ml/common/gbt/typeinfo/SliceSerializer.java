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
import org.apache.flink.ml.common.gbt.defs.Slice;

import java.io.IOException;

/** Serializer for {@link Slice}. */
public final class SliceSerializer extends TypeSerializerSingleton<Slice> {

    public static final SliceSerializer INSTANCE = new SliceSerializer();
    private static final long serialVersionUID = 1L;

    private static final SplitSerializer SPLIT_SERIALIZER = SplitSerializer.INSTANCE;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public Slice createInstance() {
        return new Slice();
    }

    @Override
    public Slice copy(Slice from) {
        Slice slice = new Slice();
        slice.start = from.start;
        slice.end = from.end;
        return slice;
    }

    @Override
    public Slice copy(Slice from, Slice reuse) {
        reuse.start = from.start;
        reuse.end = from.end;
        return reuse;
    }

    @Override
    public int getLength() {
        return 2 * IntSerializer.INSTANCE.getLength();
    }

    @Override
    public void serialize(Slice record, DataOutputView target) throws IOException {
        IntSerializer.INSTANCE.serialize(record.start, target);
        IntSerializer.INSTANCE.serialize(record.end, target);
    }

    @Override
    public Slice deserialize(DataInputView source) throws IOException {
        Slice slice = new Slice();
        slice.start = IntSerializer.INSTANCE.deserialize(source);
        slice.end = IntSerializer.INSTANCE.deserialize(source);
        return slice;
    }

    @Override
    public Slice deserialize(Slice reuse, DataInputView source) throws IOException {
        reuse.start = IntSerializer.INSTANCE.deserialize(source);
        reuse.end = IntSerializer.INSTANCE.deserialize(source);
        return reuse;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<Slice> snapshotConfiguration() {
        return new SliceSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class SliceSerializerSnapshot extends SimpleTypeSerializerSnapshot<Slice> {

        public SliceSerializerSnapshot() {
            super(SliceSerializer::new);
        }
    }
}
