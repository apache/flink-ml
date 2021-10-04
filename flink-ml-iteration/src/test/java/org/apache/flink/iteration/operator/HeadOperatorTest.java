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

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.event.CoordinatorCheckpointEvent;
import org.apache.flink.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.iteration.operator.event.SubtaskAlignedEvent;
import org.apache.flink.iteration.operator.headprocessor.RegularHeadOperatorRecordProcessor;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.checkpoint.CheckpointType;
import org.apache.flink.runtime.checkpoint.TaskStateSnapshot;
import org.apache.flink.runtime.io.network.api.CheckpointBarrier;
import org.apache.flink.runtime.io.network.api.EndOfData;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.operators.coordination.OperatorEvent;
import org.apache.flink.runtime.operators.coordination.OperatorEventGateway;
import org.apache.flink.runtime.state.CheckpointStorageLocationReference;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannel;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannelBroker;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.OneInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;
import org.apache.flink.util.FlinkException;
import org.apache.flink.util.SerializedValue;
import org.apache.flink.util.TestLogger;
import org.apache.flink.util.function.FunctionWithException;

import org.junit.Test;

import javax.annotation.Nullable;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/** Tests the {@link HeadOperator}. */
public class HeadOperatorTest extends TestLogger {

    @Test
    public void testForwardRecords() throws Exception {
        IterationID iterationId = new IterationID();
        OperatorID operatorId = new OperatorID();

        createHarnessAndRun(
                iterationId,
                operatorId,
                null,
                harness -> {
                    harness.processElement(new StreamRecord<>(IterationRecord.newRecord(1, 0), 2));
                    putFeedbackRecords(iterationId, IterationRecord.newRecord(3, 1), 3L);
                    harness.processAll();
                    harness.processElement(new StreamRecord<>(IterationRecord.newRecord(2, 0), 3));
                    putFeedbackRecords(iterationId, IterationRecord.newRecord(4, 1), 4L);
                    harness.processAll();

                    List<StreamRecord<IterationRecord<Integer>>> expectedOutput =
                            Arrays.asList(
                                    new StreamRecord<>(IterationRecord.newRecord(1, 0), 2),
                                    new StreamRecord<>(IterationRecord.newRecord(3, 1), 3),
                                    new StreamRecord<>(IterationRecord.newRecord(2, 0), 3),
                                    new StreamRecord<>(IterationRecord.newRecord(4, 1), 4));
                    assertEquals(expectedOutput, new ArrayList<>(harness.getOutput()));

                    RegularHeadOperatorRecordProcessor recordProcessor =
                            (RegularHeadOperatorRecordProcessor)
                                    RecordingHeadOperatorFactory.latestHeadOperator
                                            .getRecordProcessor();

                    assertEquals(2, (long) recordProcessor.getNumFeedbackRecordsPerEpoch().get(1));

                    return null;
                });
    }

    @Test(timeout = 60000)
    public void testSynchronizingEpochWatermark() throws Exception {
        IterationID iterationId = new IterationID();
        OperatorID operatorId = new OperatorID();

        createHarnessAndRun(
                iterationId,
                operatorId,
                null,
                harness -> {
                    harness.processElement(new StreamRecord<>(IterationRecord.newRecord(1, 0), 2));

                    // We will start a new thread to simulate the operator coordinator thread
                    CompletableFuture<Void> taskExecuteResult =
                            CompletableFuture.supplyAsync(
                                    () -> {
                                        try {
                                            RecordingOperatorEventGateway eventGateway =
                                                    (RecordingOperatorEventGateway)
                                                            RecordingHeadOperatorFactory
                                                                    .latestHeadOperator
                                                                    .getOperatorEventGateway();

                                            // We should get the aligned event for round 0 on
                                            // endInput
                                            assertNextOperatorEvent(
                                                    new SubtaskAlignedEvent(0, 0, false),
                                                    eventGateway);
                                            dispatchOperatorEvent(
                                                    harness,
                                                    operatorId,
                                                    new GloballyAlignedEvent(0, false));

                                            putFeedbackRecords(
                                                    iterationId,
                                                    IterationRecord.newRecord(4, 1),
                                                    4L);
                                            putFeedbackRecords(
                                                    iterationId,
                                                    IterationRecord.newEpochWatermark(1, "tail"),
                                                    0L);

                                            assertNextOperatorEvent(
                                                    new SubtaskAlignedEvent(1, 1, false),
                                                    eventGateway);
                                            dispatchOperatorEvent(
                                                    harness,
                                                    operatorId,
                                                    new GloballyAlignedEvent(1, true));

                                            while (RecordingHeadOperatorFactory.latestHeadOperator
                                                            .getStatus()
                                                    == HeadOperator.HeadOperatorStatus.RUNNING) {}
                                            putFeedbackRecords(
                                                    iterationId,
                                                    IterationRecord.newEpochWatermark(
                                                            Integer.MAX_VALUE + 1, "tail"),
                                                    null);

                                            return null;
                                        } catch (Throwable e) {
                                            RecordingHeadOperatorFactory.latestHeadOperator
                                                    .getMailboxExecutor()
                                                    .execute(
                                                            () -> {
                                                                throw e;
                                                            },
                                                            "poison mail");
                                            throw new CompletionException(e);
                                        }
                                    });

                    // Mark the input as finished.
                    harness.processEvent(EndOfData.INSTANCE);

                    // There should be no exception
                    taskExecuteResult.get();

                    assertEquals(
                            Arrays.asList(
                                    new StreamRecord<>(IterationRecord.newRecord(1, 0), 2),
                                    new StreamRecord<>(
                                            IterationRecord.newEpochWatermark(
                                                    0,
                                                    OperatorUtils.getUniqueSenderId(operatorId, 0)),
                                            0),
                                    new StreamRecord<>(IterationRecord.newRecord(4, 1), 4),
                                    new StreamRecord<>(
                                            IterationRecord.newEpochWatermark(
                                                    Integer.MAX_VALUE,
                                                    OperatorUtils.getUniqueSenderId(operatorId, 0)),
                                            0)),
                            new ArrayList<>(harness.getOutput()));
                    return null;
                });
    }

    @Test(timeout = 60000)
    public void testHoldCheckpointTillCoordinatorNotified() throws Exception {
        IterationID iterationId = new IterationID();
        OperatorID operatorId = new OperatorID();

        createHarnessAndRun(
                iterationId,
                operatorId,
                null,
                harness -> {
                    CompletableFuture<Void> coordinatorResult =
                            CompletableFuture.supplyAsync(
                                    () -> {
                                        try {
                                            // Slight postpone the notification
                                            Thread.sleep(2000);

                                            harness.getStreamTask()
                                                    .dispatchOperatorEvent(
                                                            operatorId,
                                                            new SerializedValue<>(
                                                                    new GloballyAlignedEvent(
                                                                            5, false)));
                                            harness.getStreamTask()
                                                    .dispatchOperatorEvent(
                                                            operatorId,
                                                            new SerializedValue<>(
                                                                    new CoordinatorCheckpointEvent(
                                                                            5)));
                                            return null;
                                        } catch (Throwable e) {
                                            RecordingHeadOperatorFactory.latestHeadOperator
                                                    .getMailboxExecutor()
                                                    .execute(
                                                            () -> {
                                                                throw e;
                                                            },
                                                            "poison mail");
                                            throw new CompletionException(e);
                                        }
                                    });

                    CheckpointBarrier barrier =
                            new CheckpointBarrier(
                                    5,
                                    5000,
                                    CheckpointOptions.alignedNoTimeout(
                                            CheckpointType.CHECKPOINT,
                                            CheckpointStorageLocationReference.getDefault()));
                    harness.processEvent(barrier);

                    // There should be no exception
                    coordinatorResult.get();

                    // If the task do not hold, it would be likely snapshot state before received
                    // the globally aligned event.
                    assertEquals(
                            Arrays.asList(
                                    new StreamRecord<>(
                                            IterationRecord.newEpochWatermark(
                                                    5,
                                                    OperatorUtils.getUniqueSenderId(operatorId, 0)),
                                            0),
                                    barrier),
                            new ArrayList<>(harness.getOutput()));
                    return null;
                });
    }

    @Test(timeout = 60000)
    public void testPostponeGloballyAlignedEventsAfterSnapshot() throws Exception {
        IterationID iterationId = new IterationID();
        OperatorID operatorId = new OperatorID();

        createHarnessAndRun(
                iterationId,
                operatorId,
                null,
                harness -> {
                    harness.getStreamTask()
                            .dispatchOperatorEvent(
                                    operatorId,
                                    new SerializedValue<>(new CoordinatorCheckpointEvent(5)));
                    harness.getStreamTask()
                            .dispatchOperatorEvent(
                                    operatorId,
                                    new SerializedValue<>(new GloballyAlignedEvent(5, false)));
                    CheckpointBarrier barrier =
                            new CheckpointBarrier(
                                    5,
                                    5000,
                                    CheckpointOptions.alignedNoTimeout(
                                            CheckpointType.CHECKPOINT,
                                            CheckpointStorageLocationReference.getDefault()));
                    harness.processEvent(barrier);
                    harness.processAll();

                    assertEquals(
                            Arrays.asList(
                                    barrier,
                                    new StreamRecord<>(
                                            IterationRecord.newEpochWatermark(
                                                    5,
                                                    OperatorUtils.getUniqueSenderId(operatorId, 0)),
                                            0)),
                            new ArrayList<>(harness.getOutput()));
                    return null;
                });
    }

    private <T> T createHarnessAndRun(
            IterationID iterationId,
            OperatorID operatorId,
            @Nullable TaskStateSnapshot snapshot,
            FunctionWithException<
                            StreamTaskMailboxTestHarness<IterationRecord<Integer>>, T, Exception>
                    runnable)
            throws Exception {
        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                OneInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .setTaskStateSnapshot(
                                1, snapshot == null ? new TaskStateSnapshot() : snapshot)
                        .setupOutputForSingletonOperatorChain(
                                new RecordingHeadOperatorFactory(
                                        iterationId,
                                        0,
                                        false,
                                        5,
                                        RecordingOperatorEventGateway::new),
                                operatorId)
                        .build()) {
            return runnable.apply(harness);
        }
    }

    private static void dispatchOperatorEvent(
            StreamTaskMailboxTestHarness<?> harness,
            OperatorID operatorId,
            OperatorEvent operatorEvent)
            throws IOException, FlinkException {
        harness.getStreamTask()
                .dispatchOperatorEvent(operatorId, new SerializedValue<>(operatorEvent));
    }

    private static void assertNextOperatorEvent(
            OperatorEvent expectedEvent, RecordingOperatorEventGateway eventGateway)
            throws InterruptedException {
        OperatorEvent nextOperatorEvent = eventGateway.operatorEvents.poll(10000, TimeUnit.SECONDS);
        assertNotNull("The expected operator event not received", nextOperatorEvent);
        assertEquals(expectedEvent, nextOperatorEvent);
    }

    private static void putFeedbackRecords(
            IterationID iterationId, IterationRecord<?> record, @Nullable Long timestamp) {
        FeedbackChannel<StreamRecord<IterationRecord<?>>> feedbackChannel =
                FeedbackChannelBroker.get()
                        .getChannel(
                                OperatorUtils.<StreamRecord<IterationRecord<?>>>createFeedbackKey(
                                                iterationId, 0)
                                        .withSubTaskIndex(0, 0));
        feedbackChannel.put(
                timestamp == null
                        ? new StreamRecord<>(record)
                        : new StreamRecord<>(record, timestamp));
    }

    private static class RecordingOperatorEventGateway implements OperatorEventGateway {

        final BlockingQueue<OperatorEvent> operatorEvents = new LinkedBlockingQueue<>();

        @Override
        public void sendEventToCoordinator(OperatorEvent operatorEvent) {
            operatorEvents.add(operatorEvent);
        }
    }

    private interface OperatorEventGatewayFactory extends Serializable {

        OperatorEventGateway create();
    }

    private static class RecordingHeadOperatorFactory extends HeadOperatorFactory {

        private final OperatorEventGatewayFactory operatorEventGatewayFactory;

        static HeadOperator latestHeadOperator;

        public RecordingHeadOperatorFactory(
                IterationID iterationId,
                int feedbackIndex,
                boolean isCriteriaStream,
                int totalHeadParallelism,
                OperatorEventGatewayFactory operatorEventGatewayFactory) {
            super(iterationId, feedbackIndex, isCriteriaStream, totalHeadParallelism);
            this.operatorEventGatewayFactory = operatorEventGatewayFactory;
        }

        @Override
        public <T extends StreamOperator<IterationRecord<?>>> T createStreamOperator(
                StreamOperatorParameters<IterationRecord<?>> streamOperatorParameters) {

            latestHeadOperator = super.createStreamOperator(streamOperatorParameters);
            return (T) latestHeadOperator;
        }

        @Override
        OperatorEventGateway createOperatorEventGateway(
                StreamOperatorParameters<IterationRecord<?>> streamOperatorParameters) {
            return operatorEventGatewayFactory.create();
        }
    }
}
