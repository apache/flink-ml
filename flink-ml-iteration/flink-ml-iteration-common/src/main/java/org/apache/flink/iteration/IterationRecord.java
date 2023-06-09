/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.iteration;

import java.util.Objects;

/** The wrapper for the records in iterative stream. */
public class IterationRecord<T> implements Cloneable {

    /** The type of iteration records. */
    public enum Type {
        RECORD,

        EPOCH_WATERMARK,

        BARRIER
    }

    private Type type;

    private int epoch;

    // -------------------------- Fields for normal records -----------------

    private T value;

    // -------------------------- Fields for epoch watermark -----------------

    /**
     * The sender is used for the receiver to distinguish the source of the event. Currently we
     * could only know the input that received the event, but there no additional information about
     * which channel it is from.
     */
    private String sender;

    // -------------------------- Fields for barrier -----------------
    private long checkpointId;

    public static <T> IterationRecord<T> newRecord(T value, int epoch) {
        return new IterationRecord<>(Type.RECORD, epoch, value, null, 0);
    }

    public static <T> IterationRecord<T> newEpochWatermark(int epoch, String sender) {
        return new IterationRecord<>(Type.EPOCH_WATERMARK, epoch, null, sender, 0);
    }

    public static <T> IterationRecord<T> newBarrier(long checkpointId) {
        return new IterationRecord<>(Type.BARRIER, 0, null, null, checkpointId);
    }

    private IterationRecord(Type type, int epoch, T value, String sender, long checkpointId) {
        this.type = type;
        this.epoch = epoch;
        this.value = value;
        this.sender = sender;
        this.checkpointId = checkpointId;
    }

    public Type getType() {
        return type;
    }

    public void setType(Type type) {
        this.type = type;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    public void incrementEpoch() {
        this.epoch += 1;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }

    public String getSender() {
        return sender;
    }

    public void setSender(String sender) {
        this.sender = sender;
    }

    public long getCheckpointId() {
        return checkpointId;
    }

    public void setCheckpointId(long checkpointId) {
        this.checkpointId = checkpointId;
    }

    @Override
    public IterationRecord<T> clone() {
        switch (type) {
            case RECORD:
                return IterationRecord.newRecord(value, epoch);
            case EPOCH_WATERMARK:
                return IterationRecord.newEpochWatermark(epoch, sender);
            case BARRIER:
                return IterationRecord.newBarrier(checkpointId);
            default:
                throw new RuntimeException("Unsupported type: " + type);
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
        IterationRecord<?> that = (IterationRecord<?>) o;
        return epoch == that.epoch
                && type == that.type
                && Objects.equals(value, that.value)
                && Objects.equals(sender, that.sender)
                && checkpointId == that.checkpointId;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, epoch, value, sender, checkpointId);
    }

    @Override
    public String toString() {
        return "IterationRecord{"
                + "type="
                + type
                + ", epoch="
                + epoch
                + ", value="
                + value
                + ", sender='"
                + sender
                + "', checkpointId="
                + checkpointId
                + "}";
    }
}
