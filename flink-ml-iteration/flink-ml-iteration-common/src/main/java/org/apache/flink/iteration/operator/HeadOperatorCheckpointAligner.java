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

package org.apache.flink.iteration.operator;

import org.apache.flink.iteration.operator.event.CoordinatorCheckpointEvent;
import org.apache.flink.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.util.function.RunnableWithException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;

import static org.apache.flink.util.Preconditions.checkState;

/**
 * Aligns the checkpoint barrier from the task inputs and the checkpoint event from the coordinator.
 * Besides, it needs to hold the other operator events after the checkpoint event till the state is
 * snapshot.
 *
 * <p>Notes that the alignment only required if its state is RUNNING. otherwise there would be no
 * more events from the coordinator.
 */
class HeadOperatorCheckpointAligner {

    private final TreeMap<Long, CheckpointAlignment> checkpointAlignmments;

    private long latestCheckpointFromCoordinator;

    private long latestAbortedCheckpoint;

    HeadOperatorCheckpointAligner() {
        this.checkpointAlignmments = new TreeMap<>();
    }

    void waitTillCoordinatorNotified(
            HeadOperator.HeadOperatorStatus status,
            long checkpointId,
            RunnableWithException defaultAction)
            throws Exception {
        CheckpointAlignment checkpointAlignment =
                checkpointAlignmments.computeIfAbsent(
                        checkpointId,
                        ignored ->
                                new CheckpointAlignment(
                                        true,
                                        status == HeadOperator.HeadOperatorStatus.RUNNING
                                                ? false
                                                : true));
        while (!checkpointAlignment.notifiedFromCoordinator) {
            defaultAction.run();
        }
        checkpointAlignment.notifiedFromChannels = true;
    }

    void coordinatorNotify(CoordinatorCheckpointEvent checkpointEvent) {
        checkState(checkpointEvent.getCheckpointId() > latestCheckpointFromCoordinator);
        latestCheckpointFromCoordinator = checkpointEvent.getCheckpointId();

        // Do nothing if later checkpoint is aborted. In this case there should not be
        // the notification from the task side.
        if (latestCheckpointFromCoordinator <= latestAbortedCheckpoint) {
            return;
        }

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

    List<GloballyAlignedEvent> onCheckpointAborted(long checkpointId) {
        if (checkpointId <= latestAbortedCheckpoint) {
            return Collections.emptyList();
        }

        latestAbortedCheckpoint = checkpointId;

        // Here we need to abort all the checkpoints <= notified checkpoint id.
        Map<Long, CheckpointAlignment> abortedAlignments =
                checkpointAlignmments.headMap(latestAbortedCheckpoint, true);
        List<GloballyAlignedEvent> events = new ArrayList<>();
        abortedAlignments
                .values()
                .forEach(alignment -> events.addAll(alignment.pendingGlobalEvents));
        abortedAlignments.clear();

        return events;
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
