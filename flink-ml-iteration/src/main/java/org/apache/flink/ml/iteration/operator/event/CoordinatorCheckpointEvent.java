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

import org.apache.flink.runtime.operators.coordination.OperatorEvent;

import java.util.Objects;

/** Coordinator received the request of checkpoints. */
public class CoordinatorCheckpointEvent implements OperatorEvent {

    private final long checkpointId;

    public CoordinatorCheckpointEvent(long checkpointId) {
        this.checkpointId = checkpointId;
    }

    public long getCheckpointId() {
        return checkpointId;
    }

    @Override
    public String toString() {
        return "CoordinatorCheckpointEvent{" + "checkpointId=" + checkpointId + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof CoordinatorCheckpointEvent)) {
            return false;
        }
        CoordinatorCheckpointEvent that = (CoordinatorCheckpointEvent) o;
        return checkpointId == that.checkpointId;
    }

    @Override
    public int hashCode() {
        return Objects.hash(checkpointId);
    }
}
