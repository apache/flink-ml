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

package org.apache.flink.ml.iteration;

import java.util.Objects;

/** The wrapper for the records in iterative stream. */
public class IterationRecord<T> implements Cloneable {

    /** The type of mini-batch stream records. */
    public enum Type {
        RECORD,

        EPOCH_WATERMARK,

        BARRIER
    }

    private Type type;

    private int round;

    // -------------------------- Fields for normal records -----------------

    private T value;

    // -------------------------- Fields for epoch watermark -----------------

    private String sender;

    // -------------------------- Fields for epoch watermark -----------------
    private long checkpointId;

    public static <T> IterationRecord<T> newRecord(T value, int round) {
        return new IterationRecord<>(Type.RECORD, round, value, null, 0);
    }

    public static <T> IterationRecord<T> newEpochWatermark(int round, String sender) {
        return new IterationRecord<>(Type.EPOCH_WATERMARK, round, null, sender, 0);
    }

    public static <T> IterationRecord<T> newBarrier(long checkpointId) {
        return new IterationRecord<>(Type.BARRIER, 0, null, null, checkpointId);
    }

    private IterationRecord(Type type, int round, T value, String sender, long checkpointId) {
        this.type = type;
        this.round = round;
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

    public int getRound() {
        return round;
    }

    public void setRound(int round) {
        this.round = round;
    }

    public void incrementalRound() {
        this.round += 1;
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
                return IterationRecord.newRecord(value, round);
            case EPOCH_WATERMARK:
                return IterationRecord.newEpochWatermark(round, sender);
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
        return round == that.round
                && type == that.type
                && Objects.equals(value, that.value)
                && Objects.equals(sender, that.sender)
                && checkpointId == that.checkpointId;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, round, value, sender, checkpointId);
    }

    @Override
    public String toString() {
        return "IterationRecord{"
                + "type="
                + type
                + ", round="
                + round
                + ", value="
                + value
                + ", sender='"
                + sender
                + "', checkpointId="
                + checkpointId
                + "}";
    }
}
