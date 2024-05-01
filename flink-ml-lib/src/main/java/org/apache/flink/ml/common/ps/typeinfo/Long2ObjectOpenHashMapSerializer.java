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

package org.apache.flink.ml.common.ps.typeinfo;

import org.apache.flink.api.common.typeutils.CompositeTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.util.Preconditions;

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

/**
 * TypeSerializer for {@link Long2ObjectOpenHashMap}.
 *
 * @param <T> The type of elements in the Long2ObjectOpenHashMap.
 */
public class Long2ObjectOpenHashMapSerializer<T> extends TypeSerializer<Long2ObjectOpenHashMap<T>> {

    private final TypeSerializer<T> elementSerializer;

    public Long2ObjectOpenHashMapSerializer(TypeSerializer<T> elementSerializer) {
        this.elementSerializer = Preconditions.checkNotNull(elementSerializer);
    }

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<Long2ObjectOpenHashMap<T>> duplicate() {
        return new Long2ObjectOpenHashMapSerializer<>(elementSerializer.duplicate());
    }

    @Override
    public Long2ObjectOpenHashMap<T> createInstance() {
        return new Long2ObjectOpenHashMap<>();
    }

    @Override
    public Long2ObjectOpenHashMap<T> copy(Long2ObjectOpenHashMap<T> from) {
        return new Long2ObjectOpenHashMap<>(from);
    }

    @Override
    public Long2ObjectOpenHashMap<T> copy(
            Long2ObjectOpenHashMap<T> from, Long2ObjectOpenHashMap<T> reuse) {
        return new Long2ObjectOpenHashMap<>(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Long2ObjectOpenHashMap<T> map, DataOutputView target) throws IOException {
        target.writeInt(map.size());
        for (Map.Entry<Long, T> entry : map.entrySet()) {
            target.writeLong(entry.getKey());
            elementSerializer.serialize(entry.getValue(), target);
        }
    }

    @Override
    public Long2ObjectOpenHashMap<T> deserialize(DataInputView source) throws IOException {
        int numEntries = source.readInt();
        Long2ObjectOpenHashMap<T> map = new Long2ObjectOpenHashMap<>(numEntries);
        for (int i = 0; i < numEntries; i++) {
            long k = source.readLong();
            T v = elementSerializer.deserialize(source);
            map.put(k, v);
        }
        return map;
    }

    @Override
    public Long2ObjectOpenHashMap<T> deserialize(
            Long2ObjectOpenHashMap<T> reuse, DataInputView source) throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        int numEntries = source.readInt();
        target.writeInt(numEntries);
        for (int i = 0; i < numEntries; ++i) {
            target.writeLong(source.readLong());
            elementSerializer.copy(source, target);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        Long2ObjectOpenHashMapSerializer<?> that = (Long2ObjectOpenHashMapSerializer<?>) o;
        return Objects.equals(elementSerializer, that.elementSerializer);
    }

    @Override
    public int hashCode() {
        return Objects.hash(elementSerializer != null ? elementSerializer.hashCode() : 0);
    }

    @Override
    public TypeSerializerSnapshot<Long2ObjectOpenHashMap<T>> snapshotConfiguration() {
        return new Long2ObjectOpenHashMapSnapshot<>(this);
    }

    private static final class Long2ObjectOpenHashMapSnapshot<T>
            extends CompositeTypeSerializerSnapshot<
                    Long2ObjectOpenHashMap<T>, Long2ObjectOpenHashMapSerializer<T>> {

        private static final int CURRENT_VERSION = 1;

        public Long2ObjectOpenHashMapSnapshot() {
            super(Long2ObjectOpenHashMapSerializer.class);
        }

        public Long2ObjectOpenHashMapSnapshot(Long2ObjectOpenHashMapSerializer<T> serializer) {
            super(serializer);
        }

        @Override
        protected int getCurrentOuterSnapshotVersion() {
            return CURRENT_VERSION;
        }

        @Override
        protected TypeSerializer<?>[] getNestedSerializers(
                Long2ObjectOpenHashMapSerializer<T> outerSerializer) {
            return new TypeSerializer[] {outerSerializer.elementSerializer};
        }

        @Override
        protected Long2ObjectOpenHashMapSerializer<T> createOuterSerializerWithNestedSerializers(
                TypeSerializer<?>[] nestedSerializers) {
            TypeSerializer<T> elementSerializer = (TypeSerializer<T>) nestedSerializers[0];
            return new Long2ObjectOpenHashMapSerializer<>(elementSerializer);
        }
    }
}
