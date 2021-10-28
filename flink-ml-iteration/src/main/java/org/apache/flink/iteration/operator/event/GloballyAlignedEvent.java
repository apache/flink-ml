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

package org.apache.flink.iteration.operator.event;

import org.apache.flink.iteration.operator.HeadOperator;
import org.apache.flink.iteration.operator.coordinator.HeadOperatorCoordinator;
import org.apache.flink.runtime.operators.coordination.OperatorEvent;

import java.util.Objects;

/**
 * The event sent from {@link HeadOperatorCoordinator} to {@link HeadOperator} to notify a round is
 * globally aligned and whether the iteration should terminate.
 */
public class GloballyAlignedEvent implements OperatorEvent {

    private final int epoch;

    private final boolean isTerminated;

    public GloballyAlignedEvent(int epoch, boolean isTerminated) {
        this.epoch = epoch;
        this.isTerminated = isTerminated;
    }

    public int getEpoch() {
        return epoch;
    }

    public boolean isTerminated() {
        return isTerminated;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        GloballyAlignedEvent that = (GloballyAlignedEvent) o;
        return epoch == that.epoch && isTerminated == that.isTerminated;
    }

    @Override
    public int hashCode() {
        return Objects.hash(epoch, isTerminated);
    }

    @Override
    public String toString() {
        return "GloballyAlignedEvent{" + "round=" + epoch + ", isTerminated=" + isTerminated + '}';
    }
}
