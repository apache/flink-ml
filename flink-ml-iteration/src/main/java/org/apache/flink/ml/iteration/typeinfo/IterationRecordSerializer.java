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

package org.apache.flink.ml.iteration.typeinfo;

import org.apache.flink.api.common.typeutils.CompositeTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.LongSerializer;
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.iteration.IterationRecord;

import java.io.IOException;
import java.util.Objects;

/** The type serializer for {@link IterationRecord}. */
public class IterationRecordSerializer<T> extends TypeSerializer<IterationRecord<T>> {
    private final TypeSerializer<T> innerSerializer;

    public IterationRecordSerializer(TypeSerializer<T> innerSerializer) {
        this.innerSerializer = innerSerializer;
    }

    public TypeSerializer<T> getInnerSerializer() {
        return innerSerializer;
    }

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<IterationRecord<T>> duplicate() {
        return new IterationRecordSerializer<>(innerSerializer.duplicate());
    }

    @Override
    public IterationRecord<T> createInstance() {
        return null;
    }

    @Override
    public IterationRecord<T> copy(IterationRecord<T> from) {
        switch (from.getType()) {
            case RECORD:
                return IterationRecord.newRecord(
                        innerSerializer.copy(from.getValue()), from.getRound());
            case EPOCH_WATERMARK:
                return IterationRecord.newEpochWatermark(from.getRound(), from.getSender());
            case BARRIER:
                return IterationRecord.newBarrier(from.getCheckpointId());
            default:
                throw new RuntimeException("Unsupported mini-batch record type " + from.getType());
        }
    }

    @Override
    public IterationRecord<T> copy(IterationRecord<T> from, IterationRecord<T> reuse) {
        from.setType(reuse.getType());
        reuse.setRound(from.getRound());

        switch (from.getType()) {
            case RECORD:
                if (reuse.getValue() != null) {
                    innerSerializer.copy(from.getValue(), reuse.getValue());
                } else {
                    reuse.setValue(innerSerializer.copy(from.getValue()));
                }
                break;
            case EPOCH_WATERMARK:
                reuse.setSender(from.getSender());
                break;
            case BARRIER:
                reuse.setCheckpointId(from.getCheckpointId());
                break;
            default:
                throw new RuntimeException("Unsupported mini-batch record type " + from.getType());
        }

        return reuse;
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(IterationRecord<T> record, DataOutputView target) throws IOException {
        // Write the mini-batch id & type, type saved as the last 2 bits of the round
        target.writeByte((byte) record.getType().ordinal());
        serializerNumber(record.getRound(), target);

        switch (record.getType()) {
            case RECORD:
                innerSerializer.serialize(record.getValue(), target);
                break;
            case EPOCH_WATERMARK:
                StringSerializer.INSTANCE.serialize(record.getSender(), target);
                break;
            case BARRIER:
                LongSerializer.INSTANCE.serialize(record.getCheckpointId(), target);
                break;
            default:
                throw new IOException("Unsupported mini-batch record type " + record.getType());
        }
    }

    @Override
    public IterationRecord<T> deserialize(DataInputView source) throws IOException {
        int type = source.readByte();
        int round = deserializeNumber(source);

        switch (IterationRecord.Type.values()[type]) {
            case RECORD:
                T value = innerSerializer.deserialize(source);
                return IterationRecord.newRecord(value, round);
            case EPOCH_WATERMARK:
                String sender = StringSerializer.INSTANCE.deserialize(source);
                return IterationRecord.newEpochWatermark(round, sender);
            case BARRIER:
                long checkpointId = LongSerializer.INSTANCE.deserialize(source);
                return IterationRecord.newBarrier(checkpointId);
            default:
                throw new IOException("Unsupported mini-batch record type " + type);
        }
    }

    @Override
    public IterationRecord<T> deserialize(IterationRecord<T> reuse, DataInputView source)
            throws IOException {
        int number = deserializeNumber(source);
        int type = number & 0x03;
        int round = number >> 2;

        reuse.setType(IterationRecord.Type.values()[type]);
        reuse.setRound(round);

        switch (reuse.getType()) {
            case RECORD:
                if (reuse.getValue() != null) {
                    innerSerializer.deserialize(reuse.getValue(), source);
                } else {
                    reuse.setValue(innerSerializer.deserialize(source));
                }
                return reuse;
            case BARRIER:
                reuse.setSender(StringSerializer.INSTANCE.deserialize(source));
                return reuse;
            default:
                throw new IOException("Unsupported mini-batch record type " + type);
        }
    }

    public void serializerNumber(int value, DataOutputView target) throws IOException {
        if (value <= 0x7F) {
            target.writeByte((byte) (value));
        } else {
            while (value > 0x7F) {
                target.writeByte((byte) ((value & 0x7F) | 0x80));
                value >>>= 7;
            }
            target.writeByte((byte) (value & 0x7F));
        }
    }

    public int deserializeNumber(DataInputView source) throws IOException {
        int offset = 0;
        int value = 0;

        byte next;
        while ((next = source.readByte()) < 0) {
            value |= (((long) (next & 0x7f)) << offset);
            offset += 7;
        }
        value |= (((long) next) << offset);

        return value;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        IterationRecord<T> record = deserialize(source);
        serialize(record, target);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        IterationRecordSerializer<?> that = (IterationRecordSerializer<?>) o;
        return Objects.equals(innerSerializer, that.innerSerializer);
    }

    @Override
    public int hashCode() {
        return innerSerializer != null ? innerSerializer.hashCode() : 0;
    }

    @Override
    public TypeSerializerSnapshot<IterationRecord<T>> snapshotConfiguration() {
        return new IterationRecordTypeSerializerSnapshot<>();
    }

    public static final class IterationRecordTypeSerializerSnapshot<T>
            extends CompositeTypeSerializerSnapshot<
                    IterationRecord<T>, IterationRecordSerializer<T>> {

        private static final int CURRENT_VERSION = 1;

        public IterationRecordTypeSerializerSnapshot() {
            super(IterationRecordSerializer.class);
        }

        @Override
        protected int getCurrentOuterSnapshotVersion() {
            return CURRENT_VERSION;
        }

        @Override
        protected TypeSerializer<?>[] getNestedSerializers(
                IterationRecordSerializer<T> tIterationRecordSerializer) {
            return new TypeSerializer[] {tIterationRecordSerializer.getInnerSerializer()};
        }

        @Override
        protected IterationRecordSerializer<T> createOuterSerializerWithNestedSerializers(
                TypeSerializer<?>[] typeSerializers) {
            TypeSerializer<T> elementSerializer = (TypeSerializer<T>) typeSerializers[0];
            return new IterationRecordSerializer<>(elementSerializer);
        }
    }
}
