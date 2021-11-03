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
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.iteration.typeinfo.IterationRecordSerializer;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.runtime.checkpoint.CheckpointMetaData;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.checkpoint.CheckpointType;
import org.apache.flink.runtime.checkpoint.TaskStateSnapshot;
import org.apache.flink.runtime.io.network.api.CheckpointBarrier;
import org.apache.flink.runtime.io.network.api.writer.RecordOrEventCollectingResultPartitionWriter;
import org.apache.flink.runtime.io.network.api.writer.ResultPartitionWriter;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.CheckpointStorageLocationReference;
import org.apache.flink.streaming.runtime.streamrecord.StreamElement;
import org.apache.flink.streaming.runtime.streamrecord.StreamElementSerializer;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;
import org.apache.flink.streaming.runtime.tasks.TwoInputStreamTask;
import org.apache.flink.util.TestLogger;
import org.apache.flink.util.function.FunctionWithException;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import javax.annotation.Nullable;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/** Test the behavior of {@link ReplayOperator}. */
public class ReplayOperatorTest extends TestLogger {

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test(timeout = 60000)
    public void testReplaying() throws Exception {
        final int numRecords = 10;
        OperatorID operatorId = new OperatorID();

        createHarnessAndRun(
                operatorId,
                null,
                harness -> {
                    // First round
                    for (int i = 0; i < numRecords; ++i) {
                        harness.processElement(
                                new StreamRecord<>(IterationRecord.newRecord(i, 0)), 0, 0);
                    }
                    harness.endInput(0, true);
                    harness.processElement(
                            new StreamRecord<>(IterationRecord.newEpochWatermark(0, "sender1")),
                            1,
                            0);
                    assertOutputAllRecordsAndEpochWatermark(
                            harness.getOutput(), numRecords, operatorId, 0);
                    harness.getOutput().clear();

                    // The round 1
                    harness.processElement(
                            new StreamRecord<>(IterationRecord.newEpochWatermark(1, "sender1")),
                            1,
                            0);
                    // The output would be done asynchronously inside the ReplayerOperator.
                    while (harness.getOutput().size() < numRecords + 1) {
                        Thread.sleep(500);
                    }
                    assertOutputAllRecordsAndEpochWatermark(
                            harness.getOutput(), numRecords, operatorId, 1);
                    harness.getOutput().clear();

                    // The round 2
                    harness.processElement(
                            new StreamRecord<>(IterationRecord.newEpochWatermark(2, "sender1")),
                            1,
                            0);
                    // The output would be done asynchronously inside the ReplayerOperator.
                    while (harness.getOutput().size() < numRecords + 1) {
                        Thread.sleep(500);
                    }
                    assertOutputAllRecordsAndEpochWatermark(
                            harness.getOutput(), numRecords, operatorId, 2);
                    return null;
                });
    }

    @Test
    public void testSnapshotAndRestoreOnFirstEpoch() throws Exception {
        final int numRecords = 10;
        OperatorID operatorId = new OperatorID();

        List<Object> firstRoundOutput = new ArrayList<>();
        List<Object> secondRoundOutput = new ArrayList<>();

        TaskStateSnapshot snapshot =
                createHarnessAndRun(
                        operatorId,
                        null,
                        harness -> {
                            harness.getTaskStateManager().getWaitForReportLatch().reset();

                            for (int i = 0; i < numRecords / 2; ++i) {
                                harness.processElement(
                                        new StreamRecord<>(IterationRecord.newRecord(i, 0)), 0, 0);
                            }

                            harness.getStreamTask()
                                    .triggerCheckpointAsync(
                                            new CheckpointMetaData(2, 1000),
                                            CheckpointOptions.alignedNoTimeout(
                                                    CheckpointType.CHECKPOINT,
                                                    CheckpointStorageLocationReference
                                                            .getDefault()));
                            harness.processAll();

                            firstRoundOutput.addAll(harness.getOutput());

                            harness.getTaskStateManager().getWaitForReportLatch().await();
                            return harness.getTaskStateManager()
                                    .getLastJobManagerTaskStateSnapshot();
                        });

        createHarnessAndRun(
                operatorId,
                snapshot,
                harness -> {
                    for (int i = numRecords / 2; i < numRecords; ++i) {
                        harness.processElement(
                                new StreamRecord<>(IterationRecord.newRecord(i, 0)), 0, 0);
                    }
                    harness.endInput(0, true);
                    harness.processElement(
                            new StreamRecord<>(IterationRecord.newEpochWatermark(0, "send-0")),
                            1,
                            0);
                    harness.processAll();
                    firstRoundOutput.addAll(harness.getOutput());

                    // The second round
                    harness.getOutput().clear();
                    harness.processElement(
                            new StreamRecord<>(IterationRecord.newEpochWatermark(1, "send-0")),
                            1,
                            0);
                    secondRoundOutput.addAll(harness.getOutput());

                    return null;
                });

        assertOutputAllRecordsAndEpochWatermark(firstRoundOutput, numRecords, operatorId, 0);
        assertOutputAllRecordsAndEpochWatermark(secondRoundOutput, numRecords, operatorId, 1);
    }

    @Test
    public void testSnapshotAndRestoreOnSecondEpoch() throws Exception {
        final int numRecords = 10;
        OperatorID operatorId = new OperatorID();

        List<Object> firstRoundOutput = new ArrayList<>();
        List<Object> secondRoundOutput = new ArrayList<>();

        HookableOutput hookableOutput = new HookableOutput(numRecords + numRecords / 2);
        TaskStateSnapshot snapshot =
                createHarnessAndRun(
                        operatorId,
                        null,
                        harness -> {
                            harness.getTaskStateManager().getWaitForReportLatch().reset();

                            for (int i = 0; i < numRecords; ++i) {
                                harness.processElement(
                                        new StreamRecord<>(IterationRecord.newRecord(i, 0)), 0, 0);
                            }
                            harness.endInput(0, true);
                            harness.processElement(
                                    new StreamRecord<>(
                                            IterationRecord.newEpochWatermark(0, "send-0")),
                                    1,
                                    0);
                            harness.processAll();
                            firstRoundOutput.addAll(harness.getOutput());

                            harness.getOutput().clear();
                            harness.getTaskStateManager().getWaitForReportLatch().reset();
                            // We have to simulate another thread insert checkpoint barrier
                            hookableOutput.setRunnable(
                                    () ->
                                            harness.getStreamTask()
                                                    .triggerCheckpointAsync(
                                                            new CheckpointMetaData(2, 1000),
                                                            CheckpointOptions.alignedNoTimeout(
                                                                    CheckpointType.CHECKPOINT,
                                                                    CheckpointStorageLocationReference
                                                                            .getDefault())));
                            // Slightly postpone the epoch watermark.
                            harness.processElement(
                                    new StreamRecord<>(
                                            IterationRecord.newEpochWatermark(1, "send-0")),
                                    1,
                                    0);
                            harness.processAll();
                            secondRoundOutput.addAll(harness.getOutput());

                            harness.getTaskStateManager().getWaitForReportLatch().await();
                            return harness.getTaskStateManager()
                                    .getLastJobManagerTaskStateSnapshot();
                        },
                        hookableOutput);

        createHarnessAndRun(
                operatorId,
                snapshot,
                harness -> {
                    // In this case, we expected the input would finish immediately.
                    harness.endInput(0, true);
                    secondRoundOutput.addAll(harness.getOutput());
                    return null;
                });

        assertOutputAllRecordsAndEpochWatermark(firstRoundOutput, numRecords, operatorId, 0);
        assertOutputAllRecordsAndEpochWatermark(secondRoundOutput, numRecords, operatorId, 1);
    }

    private <T> T createHarnessAndRun(
            OperatorID operatorId,
            @Nullable TaskStateSnapshot snapshot,
            FunctionWithException<
                            StreamTaskMailboxTestHarness<IterationRecord<Integer>>, T, Exception>
                    runnable,
            ResultPartitionWriter... additionalOutput)
            throws Exception {
        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                TwoInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO), 1)
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.VOID_TYPE_INFO), 1)
                        .addAdditionalOutput(additionalOutput)
                        .setTaskStateSnapshot(
                                1, snapshot == null ? new TaskStateSnapshot() : snapshot)
                        .setupOutputForSingletonOperatorChain(new ReplayOperator<>(), operatorId)
                        .buildUnrestored()) {
            harness.getStreamTask()
                    .getEnvironment()
                    .getTaskManagerInfo()
                    .getConfiguration()
                    .set(
                            IterationOptions.DATA_CACHE_PATH,
                            "file://" + tempFolder.newFolder().getAbsolutePath());
            harness.getStreamTask().restore();
            return runnable.apply(harness);
        }
    }

    private void assertOutputAllRecordsAndEpochWatermark(
            Collection<Object> output, int numRecords, OperatorID operatorId, int round) {
        assertEquals(
                Stream.concat(
                                IntStream.range(0, numRecords)
                                        .boxed()
                                        .map(
                                                i ->
                                                        new StreamRecord<>(
                                                                IterationRecord.newRecord(
                                                                        i, round))),
                                Stream.of(
                                        new StreamRecord<>(
                                                IterationRecord.newEpochWatermark(
                                                        round,
                                                        OperatorUtils.getUniqueSenderId(
                                                                operatorId, 0)))))
                        .collect(Collectors.toList()),
                output.stream()
                        .filter(e -> e.getClass() != CheckpointBarrier.class)
                        .collect(Collectors.toList()));
    }

    private static class HookableOutput
            extends RecordOrEventCollectingResultPartitionWriter<StreamElement> {

        private int remainingRecordsToWait;

        @Nullable private Runnable runnable;

        public HookableOutput(int triggerCount) {
            super(
                    new ArrayDeque<>(),
                    new StreamElementSerializer<>(
                            new IterationRecordSerializer<>(IntSerializer.INSTANCE)));
            this.remainingRecordsToWait = triggerCount;
        }

        public void setRunnable(Runnable runnable) {
            this.runnable = runnable;
        }

        @Override
        public void emitRecord(ByteBuffer record, int targetSubpartition) throws IOException {
            super.emitRecord(record, targetSubpartition);
            tryTrigger();
        }

        @Override
        public void broadcastRecord(ByteBuffer record) throws IOException {
            super.broadcastRecord(record);
            tryTrigger();
        }

        private void tryTrigger() {
            if (remainingRecordsToWait > 0) {
                remainingRecordsToWait--;
                if (remainingRecordsToWait == 0 && runnable != null) {
                    runnable.run();
                }
            }
        }
    }
}
