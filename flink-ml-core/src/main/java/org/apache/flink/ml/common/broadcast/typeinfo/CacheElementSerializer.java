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

package org.apache.flink.ml.common.broadcast.typeinfo;

import org.apache.flink.api.common.typeutils.CompositeTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.LongSerializer;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.broadcast.typeinfo.CacheElement.Type;

import java.io.IOException;
import java.util.Objects;

/**
 * TypeSerializer for {@link CacheElement}.
 *
 * @param <T> the record type.
 */
public class CacheElementSerializer<T> extends TypeSerializer<CacheElement<T>> {

    private final TypeSerializer<T> recordSerializer;

    public CacheElementSerializer(TypeSerializer<T> recordSerializer) {
        this.recordSerializer = recordSerializer;
    }

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<CacheElement<T>> duplicate() {
        return new CacheElementSerializer<>(recordSerializer.duplicate());
    }

    @Override
    public CacheElement<T> createInstance() {
        return null;
    }

    @Override
    public CacheElement<T> copy(CacheElement<T> from) {
        switch (from.getType()) {
            case RECORD:
                return CacheElement.newRecord(recordSerializer.copy(from.getRecord()));
            case WATERMARK:
                return CacheElement.newWatermark(from.getWatermark());
            default:
                throw new RuntimeException(
                        "Unsupported Record or Watermark type " + from.getType());
        }
    }

    @Override
    public CacheElement<T> copy(CacheElement<T> from, CacheElement<T> reuse) {
        switch (from.getType()) {
            case RECORD:
                if (reuse.getRecord() != null) {
                    recordSerializer.copy(from.getRecord(), reuse.getRecord());
                } else {
                    reuse.setRecord(recordSerializer.copy(from.getRecord()));
                }
                break;
            case WATERMARK:
                reuse.setWatermark(from.getWatermark());
                break;
            default:
                throw new RuntimeException(
                        "Unsupported Record or Watermark type " + from.getType());
        }

        return reuse;
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(CacheElement<T> record, DataOutputView target) throws IOException {
        target.writeByte((byte) record.getType().ordinal());

        switch (record.getType()) {
            case RECORD:
                recordSerializer.serialize(record.getRecord(), target);
                break;
            case WATERMARK:
                LongSerializer.INSTANCE.serialize(record.getWatermark(), target);
                break;
            default:
                throw new RuntimeException(
                        "Unsupported Record or Watermark type " + record.getType());
        }
    }

    @Override
    public CacheElement<T> deserialize(DataInputView source) throws IOException {
        int type = source.readByte();
        switch (CacheElement.Type.values()[type]) {
            case RECORD:
                T value = recordSerializer.deserialize(source);
                return CacheElement.newRecord(value);
            case WATERMARK:
                long watermark = LongSerializer.INSTANCE.deserialize(source);
                return CacheElement.newWatermark(watermark);
            default:
                throw new RuntimeException("Unsupported Record or Watermark type " + type);
        }
    }

    @Override
    public CacheElement<T> deserialize(CacheElement<T> reuse, DataInputView source)
            throws IOException {
        int type = source.readByte();
        switch (CacheElement.Type.values()[type]) {
            case RECORD:
                reuse.setType(Type.RECORD);
                reuse.setRecord(recordSerializer.deserialize(source));
                break;
            case WATERMARK:
                reuse.setType(Type.WATERMARK);
                reuse.setWatermark(LongSerializer.INSTANCE.deserialize(source));
                break;
            default:
                throw new RuntimeException("Unsupported Record or Watermark type " + type);
        }
        return reuse;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        CacheElement<T> cacheElement = deserialize(source);
        serialize(cacheElement, target);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        CacheElementSerializer<?> that = (CacheElementSerializer<?>) obj;
        return Objects.equals(recordSerializer, that.recordSerializer);
    }

    @Override
    public int hashCode() {
        return recordSerializer != null ? recordSerializer.hashCode() : 0;
    }

    @Override
    public TypeSerializerSnapshot<CacheElement<T>> snapshotConfiguration() {
        return new CacheElementSerializerSnapshot<>();
    }

    /** The serializer snapshot class for {@link CacheElementSerializer}. */
    private static final class CacheElementSerializerSnapshot<T>
            extends CompositeTypeSerializerSnapshot<CacheElement<T>, CacheElementSerializer<T>> {

        private static final int CURRENT_VERSION = 1;

        public CacheElementSerializerSnapshot() {
            super(CacheElementSerializer.class);
        }

        @Override
        protected int getCurrentOuterSnapshotVersion() {
            return CURRENT_VERSION;
        }

        @Override
        protected TypeSerializer<?>[] getNestedSerializers(
                CacheElementSerializer<T> tIterationRecordSerializer) {
            return new TypeSerializer[] {tIterationRecordSerializer.recordSerializer};
        }

        @Override
        @SuppressWarnings("unchecked")
        protected CacheElementSerializer<T> createOuterSerializerWithNestedSerializers(
                TypeSerializer<?>[] typeSerializers) {
            TypeSerializer<T> elementSerializer = (TypeSerializer<T>) typeSerializers[0];
            return new CacheElementSerializer<>(elementSerializer);
        }
    }
}
