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
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;
import org.apache.flink.streaming.runtime.tasks.TwoInputStreamTask;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Queue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/** Test the behavior of {@link ReplayOperator}. */
public class ReplayOperatorTest {

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test(timeout = 60000)
    public void testReplaying() throws Exception {
        final int numRecords = 10;
        OperatorID operatorId = new OperatorID();

        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                TwoInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO), 1)
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.VOID_TYPE_INFO), 1)
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

            // First round
            for (int i = 0; i < numRecords; ++i) {
                harness.processElement(new StreamRecord<>(IterationRecord.newRecord(i, 0)), 0, 0);
            }
            harness.endInput(0, true);
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(0, "sender1")), 1, 0);
            assertOutputAllRecordsAndEpochWatermark(harness.getOutput(), numRecords, operatorId, 0);
            harness.getOutput().clear();

            // The round 1
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(1, "sender1")), 1, 0);
            // The output would be done asynchronously inside the ReplayerOperator.
            while (harness.getOutput().size() < numRecords + 1) {
                Thread.sleep(500);
            }
            assertOutputAllRecordsAndEpochWatermark(harness.getOutput(), numRecords, operatorId, 1);
            harness.getOutput().clear();

            // The round 2
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(2, "sender1")), 1, 0);
            // The output would be done asynchronously inside the ReplayerOperator.
            while (harness.getOutput().size() < numRecords + 1) {
                Thread.sleep(500);
            }
            assertOutputAllRecordsAndEpochWatermark(harness.getOutput(), numRecords, operatorId, 2);
        }
    }

    private void assertOutputAllRecordsAndEpochWatermark(
            Queue<Object> output, int numRecords, OperatorID operatorId, int round) {
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
                new ArrayList<>(output));
    }
}
