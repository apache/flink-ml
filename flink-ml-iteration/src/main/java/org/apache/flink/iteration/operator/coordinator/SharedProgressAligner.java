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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.iteration.operator.event.SubtaskAlignedEvent;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.jobgraph.OperatorInstanceID;
import org.apache.flink.runtime.operators.coordination.OperatorCoordinator;
import org.apache.flink.util.ExceptionUtils;
import org.apache.flink.util.function.ThrowingRunnable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.apache.flink.util.Preconditions.checkState;

/**
 * The progress aligner shared between multiple {@link HeadOperatorCoordinator}. It maintains the
 * information for each round, once one round is aligned, it would notify all the register
 * consumers.
 */
public class SharedProgressAligner {

    private static final Logger LOG = LoggerFactory.getLogger(SharedProgressAligner.class);

    public static ConcurrentHashMap<IterationID, SharedProgressAligner> instances =
            new ConcurrentHashMap<>();

    private final IterationID iterationId;

    private final int totalHeadParallelism;

    private final OperatorCoordinator.Context context;

    private final Executor executor;

    private final Map<Integer, EpochStatus> statusByEpoch;

    private final Map<OperatorID, Consumer<GloballyAlignedEvent>> alignedConsumers;

    public static SharedProgressAligner getOrCreate(
            IterationID iterationId,
            int totalHeadParallelism,
            OperatorCoordinator.Context context,
            Supplier<Executor> executorFactory) {
        return instances.computeIfAbsent(
                iterationId,
                ignored ->
                        new SharedProgressAligner(
                                iterationId, totalHeadParallelism, context, executorFactory.get()));
    }

    @VisibleForTesting
    static ConcurrentHashMap<IterationID, SharedProgressAligner> getInstances() {
        return instances;
    }

    private SharedProgressAligner(
            IterationID iterationId,
            int totalHeadParallelism,
            OperatorCoordinator.Context context,
            Executor executor) {
        this.iterationId = Objects.requireNonNull(iterationId);
        this.totalHeadParallelism = totalHeadParallelism;
        this.context = Objects.requireNonNull(context);
        this.executor = Objects.requireNonNull(executor);

        this.statusByEpoch = new HashMap<>();
        this.alignedConsumers = new HashMap<>();
    }

    public void registerAlignedConsumer(
            OperatorID operatorID, Consumer<GloballyAlignedEvent> alignedConsumer) {
        runInEventLoop(
                () -> this.alignedConsumers.put(operatorID, alignedConsumer),
                "Register consumer %s",
                operatorID.toHexString());
    }

    public void unregisterConsumer(OperatorID operatorID) {
        synchronized (this) {
            runInEventLoop(
                    () -> {
                        this.alignedConsumers.remove(operatorID);
                        if (alignedConsumers.isEmpty()) {
                            instances.remove(iterationId);
                        }
                    },
                    "Unregister consumer %s",
                    operatorID.toHexString());
        }
    }

    public void reportSubtaskProgress(
            OperatorID operatorId, int subtaskIndex, SubtaskAlignedEvent subtaskAlignedEvent) {
        runInEventLoop(
                () -> {
                    LOG.debug(
                            "Processing {} from {}-{}",
                            subtaskAlignedEvent,
                            operatorId,
                            subtaskIndex);
                    EpochStatus roundStatus =
                            statusByEpoch.computeIfAbsent(
                                    subtaskAlignedEvent.getEpoch(),
                                    round -> new EpochStatus(round, totalHeadParallelism));
                    boolean globallyAligned =
                            roundStatus.report(operatorId, subtaskIndex, subtaskAlignedEvent);
                    if (globallyAligned) {
                        GloballyAlignedEvent globallyAlignedEvent =
                                new GloballyAlignedEvent(
                                        subtaskAlignedEvent.getEpoch(), roundStatus.isTerminated());
                        for (Consumer<GloballyAlignedEvent> consumer : alignedConsumers.values()) {
                            consumer.accept(globallyAlignedEvent);
                        }
                    }
                },
                "Report subtask %s-%d",
                operatorId.toHexString(),
                subtaskIndex);
    }

    private void runInEventLoop(
            ThrowingRunnable<Throwable> action,
            String actionName,
            Object... actionNameFormatParameters) {
        executor.execute(
                () -> {
                    try {
                        action.run();
                    } catch (Throwable t) {
                        ExceptionUtils.rethrowIfFatalErrorOrOOM(t);

                        String actionString = String.format(actionName, actionNameFormatParameters);
                        LOG.error(
                                "Uncaught exception in the SharedProgressAligner for iteration {} while {}. Triggering job failover.",
                                iterationId,
                                actionString,
                                t);
                        context.failJob(t);
                    }
                });
    }

    @VisibleForTesting
    int getNumberConsumers() {
        return alignedConsumers.size();
    }

    private static class EpochStatus {

        private final int epoch;

        private final long totalHeadParallelism;

        private final Map<OperatorInstanceID, SubtaskAlignedEvent> reportedSubtasks;

        public EpochStatus(int epoch, long totalHeadParallelism) {
            this.epoch = epoch;
            this.totalHeadParallelism = totalHeadParallelism;
            this.reportedSubtasks = new HashMap<>();
        }

        public boolean report(OperatorID operatorID, int subtaskIndex, SubtaskAlignedEvent event) {
            reportedSubtasks.put(new OperatorInstanceID(subtaskIndex, operatorID), event);
            checkState(
                    reportedSubtasks.size() <= totalHeadParallelism,
                    "Received more subtasks"
                            + reportedSubtasks
                            + "than the expected total parallelism "
                            + totalHeadParallelism);
            return reportedSubtasks.size() == totalHeadParallelism;
        }

        public boolean isTerminated() {
            checkState(
                    reportedSubtasks.size() == totalHeadParallelism,
                    "The round is not globally aligned yet");

            // We never terminate at round 0.
            if (epoch == 0) {
                return false;
            }

            long totalRecord = 0;
            boolean hasCriteriaStream = false;
            long totalCriteriaRecord = 0;

            for (SubtaskAlignedEvent event : reportedSubtasks.values()) {
                totalRecord += event.getNumRecordsThisRound();
                if (event.isCriteriaStream()) {
                    hasCriteriaStream = true;
                    totalCriteriaRecord += event.getNumRecordsThisRound();
                }
            }

            return totalRecord == 0 || (hasCriteriaStream && totalCriteriaRecord == 0);
        }
    }
}
