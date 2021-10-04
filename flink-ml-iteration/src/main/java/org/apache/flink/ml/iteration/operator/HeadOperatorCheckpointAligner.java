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

package org.apache.flink.ml.iteration.operator;

import org.apache.flink.ml.iteration.operator.event.CoordinatorCheckpointEvent;
import org.apache.flink.ml.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.util.function.RunnableWithException;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.TreeMap;

import static org.apache.flink.util.Preconditions.checkState;

/**
 * Aligns the checkpoint barrier from the task inputs and the checkpoint event from the coordinator.
 * Besides, it needs to hold the other operator events after the checkpoint event till the state is
 * snapshot.
 */
class HeadOperatorCheckpointAligner {

    private final TreeMap<Long, CheckpointAlignment> checkpointAlignmments;

    private long latestCheckpointFromCoordinator;

    HeadOperatorCheckpointAligner() {
        this.checkpointAlignmments = new TreeMap<>();
    }

    void waitTillCoordinatorNotified(long checkpointId, RunnableWithException defaultAction)
            throws Exception {
        CheckpointAlignment checkpointAlignment =
                checkpointAlignmments.computeIfAbsent(
                        checkpointId, ignored -> new CheckpointAlignment(true, false));
        while (!checkpointAlignment.notifiedFromCoordinator) {
            defaultAction.run();
        }
        checkpointAlignment.notifiedFromChannels = true;
    }

    void coordinatorNotify(CoordinatorCheckpointEvent checkpointEvent) {
        checkState(checkpointEvent.getCheckpointId() > latestCheckpointFromCoordinator);
        latestCheckpointFromCoordinator = checkpointEvent.getCheckpointId();
        CheckpointAlignment checkpointAlignment =
                checkpointAlignmments.computeIfAbsent(
                        checkpointEvent.getCheckpointId(),
                        ignored -> new CheckpointAlignment(false, true));
        checkpointAlignment.notifiedFromCoordinator = true;
    }

    Optional<GloballyAlignedEvent> checkHoldingGloballyAlignedEvent(
            GloballyAlignedEvent globallyAlignedEvent) {
        CheckpointAlignment checkpointAlignment =
                checkpointAlignmments.get(latestCheckpointFromCoordinator);
        if (checkpointAlignment != null && !checkpointAlignment.notifiedFromChannels) {
            checkpointAlignment.pendingGlobalEvents.add(globallyAlignedEvent);
            return Optional.empty();
        }

        return Optional.of(globallyAlignedEvent);
    }

    List<GloballyAlignedEvent> onStateSnapshot(long checkpointId) {
        CheckpointAlignment checkpointAlignment = checkpointAlignmments.remove(checkpointId);
        checkState(
                checkpointAlignment.notifiedFromCoordinator
                        && checkpointAlignment.notifiedFromChannels,
                "Checkpoint " + checkpointId + " is not fully aligned");
        return checkpointAlignment.pendingGlobalEvents;
    }

    private static class CheckpointAlignment {

        final List<GloballyAlignedEvent> pendingGlobalEvents;

        boolean notifiedFromChannels;

        boolean notifiedFromCoordinator;

        public CheckpointAlignment(boolean notifiedFromChannels, boolean notifiedFromCoordinator) {
            this.pendingGlobalEvents = new ArrayList<>();

            this.notifiedFromChannels = notifiedFromChannels;
            this.notifiedFromCoordinator = notifiedFromCoordinator;
        }
    }
}
