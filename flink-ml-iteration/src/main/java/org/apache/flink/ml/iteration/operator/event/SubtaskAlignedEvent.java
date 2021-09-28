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

package org.apache.flink.ml.iteration.operator.event;

import org.apache.flink.ml.iteration.operator.HeadOperator;
import org.apache.flink.ml.iteration.operator.coordinator.HeadOperatorCoordinator;
import org.apache.flink.runtime.operators.coordination.OperatorEvent;

import java.util.Objects;

/**
 * The event sent from {@link HeadOperator} to {@link HeadOperatorCoordinator} to notify the subtask
 * has received the EpochWatermark for the specified round.
 */
public class SubtaskAlignedEvent implements OperatorEvent {

    private final int round;

    private final long numRecordsThisRound;

    private final boolean isCriteriaStream;

    public SubtaskAlignedEvent(int round, long numRecordsThisRound, boolean isCriteriaStream) {
        this.round = round;
        this.numRecordsThisRound = numRecordsThisRound;
        this.isCriteriaStream = isCriteriaStream;
    }

    public int getRound() {
        return round;
    }

    public long getNumRecordsThisRound() {
        return numRecordsThisRound;
    }

    public boolean isCriteriaStream() {
        return isCriteriaStream;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        SubtaskAlignedEvent that = (SubtaskAlignedEvent) o;
        return round == that.round
                && numRecordsThisRound == that.numRecordsThisRound
                && isCriteriaStream == that.isCriteriaStream;
    }

    @Override
    public int hashCode() {
        return Objects.hash(round, numRecordsThisRound, isCriteriaStream);
    }

    @Override
    public String toString() {
        return "SubtaskAlignedEvent{"
                + "round="
                + round
                + ", numRecordsThisRound="
                + numRecordsThisRound
                + ", isCriteriaStream="
                + isCriteriaStream
                + '}';
    }
}
