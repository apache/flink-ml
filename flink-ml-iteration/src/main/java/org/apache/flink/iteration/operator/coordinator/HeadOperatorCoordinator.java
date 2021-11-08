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

package org.apache.flink.iteration.operator.coordinator;

import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.operator.HeadOperator;
import org.apache.flink.iteration.operator.event.CoordinatorCheckpointEvent;
import org.apache.flink.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.iteration.operator.event.SubtaskAlignedEvent;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.operators.coordination.OperatorCoordinator;
import org.apache.flink.runtime.operators.coordination.OperatorEvent;

import javax.annotation.Nullable;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;

/**
 * The OperatorCoordinator of the {@link HeadOperator}, it notifies the {@link
 * SharedProgressAligner} when received aligned event from the operator, and emit the globally
 * aligned event back after one round is globally aligned.
 */
public class HeadOperatorCoordinator implements OperatorCoordinator, SharedProgressAlignerListener {

    private final Context context;

    private final SubtaskGateway[] subtaskGateways;

    private final SharedProgressAligner sharedProgressAligner;

    public HeadOperatorCoordinator(Context context, SharedProgressAligner sharedProgressAligner) {
        this.context = context;
        this.sharedProgressAligner = Objects.requireNonNull(sharedProgressAligner);
        this.subtaskGateways = new SubtaskGateway[context.currentParallelism()];

        sharedProgressAligner.registerAlignedListener(context.getOperatorId(), this);
    }

    @Override
    public void start() {}

    @Override
    public void subtaskReady(int subtaskIndex, SubtaskGateway subtaskGateway) {
        this.subtaskGateways[subtaskIndex] = subtaskGateway;
    }

    @Override
    public void resetToCheckpoint(long checkpointId, @Nullable byte[] bytes) {
        for (int i = 0; i < context.currentParallelism(); ++i) {
            sharedProgressAligner.removeProgressInfo(context.getOperatorId());
        }
    }

    @Override
    public void subtaskFailed(int subtaskIndex, @Nullable Throwable throwable) {
        sharedProgressAligner.removeProgressInfo(context.getOperatorId(), subtaskIndex);
    }

    @Override
    public void handleEventFromOperator(int subtaskIndex, OperatorEvent operatorEvent) {
        if (operatorEvent instanceof SubtaskAlignedEvent) {
            sharedProgressAligner.reportSubtaskProgress(
                    context.getOperatorId(), subtaskIndex, (SubtaskAlignedEvent) operatorEvent);
        } else {
            throw new UnsupportedOperationException("Not supported event: " + operatorEvent);
        }
    }

    @Override
    public void checkpointCoordinator(long l, CompletableFuture<byte[]> completableFuture) {
        sharedProgressAligner.requestCheckpoint(l, context.currentParallelism(), completableFuture);
    }

    public void onAligned(GloballyAlignedEvent globallyAlignedEvent) {
        for (int i = 0; i < context.currentParallelism(); ++i) {
            subtaskGateways[i].sendEvent(globallyAlignedEvent);
        }
    }

    @Override
    public void onCheckpointAligned(CoordinatorCheckpointEvent coordinatorCheckpointEvent) {
        for (int i = 0; i < context.currentParallelism(); ++i) {
            subtaskGateways[i].sendEvent(coordinatorCheckpointEvent);
        }
    }

    @Override
    public void close() {
        sharedProgressAligner.unregisterListener(context.getOperatorId());
    }

    @Override
    public void notifyCheckpointComplete(long l) {}

    @Override
    public void subtaskReset(int i, long l) {}

    /** The factory of {@link HeadOperatorCoordinator}. */
    public static class HeadOperatorCoordinatorProvider implements Provider {

        private final OperatorID operatorId;

        private final IterationID iterationId;

        private final int totalHeadParallelism;

        public HeadOperatorCoordinatorProvider(
                OperatorID operatorId, IterationID iterationId, int totalHeadParallelism) {
            this.operatorId = operatorId;
            this.iterationId = iterationId;
            this.totalHeadParallelism = totalHeadParallelism;
        }

        @Override
        public OperatorID getOperatorId() {
            return operatorId;
        }

        @Override
        public OperatorCoordinator create(Context context) {
            SharedProgressAligner sharedProgressAligner =
                    SharedProgressAligner.getOrCreate(
                            iterationId,
                            totalHeadParallelism,
                            context,
                            () ->
                                    Executors.newSingleThreadScheduledExecutor(
                                            runnable -> {
                                                Thread thread = new Thread(runnable);
                                                thread.setName(
                                                        "SharedProgressAligner-" + iterationId);
                                                return thread;
                                            }));
            return new HeadOperatorCoordinator(context, sharedProgressAligner);
        }
    }
}
