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

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.feedback.SpillableFeedbackChannel;
import org.apache.flink.iteration.operator.feedback.SpillableFeedbackChannelBroker;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.runtime.memory.MemoryAllocationException;
import org.apache.flink.runtime.testutils.DirectScheduledExecutorService;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;
import org.apache.flink.util.TestLogger;

import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/** Tests the {@link TailOperator}. */
public class TailOperatorTest extends TestLogger {

    @Test
    public void testIncrementRoundWithoutObjectReuse() throws Exception {
        IterationID iterationId = new IterationID();

        IterationRecordTypeInfo typeInfo =
                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO);
        TypeSerializer serializer = typeInfo.createSerializer(new ExecutionConfig());
        OneInputStreamOperatorTestHarness<IterationRecord<?>, Void> testHarness =
                new OneInputStreamOperatorTestHarness<>(
                        new TailOperator(iterationId, 0), serializer);
        testHarness.open();
        SpillableFeedbackChannel channel =
                initializeFeedbackChannel(testHarness.getOperator(), iterationId, 0, 0, 0);

        testHarness.processElement(IterationRecord.newRecord(1, 1), 2);
        testHarness.processElement(IterationRecord.newRecord(2, 1), 3);
        testHarness.processElement(IterationRecord.newEpochWatermark(2, "sender1"), 4);

        List<StreamRecord<IterationRecord<?>>> iterationRecords = getFeedbackRecords(channel);
        assertEquals(
                Arrays.asList(
                        new StreamRecord<>(IterationRecord.newRecord(1, 2), 2),
                        new StreamRecord<>(IterationRecord.newRecord(2, 2), 3),
                        new StreamRecord<>(IterationRecord.newEpochWatermark(3, "sender1"), 4)),
                iterationRecords);
    }

    @Test
    public void testIncrementRoundWithObjectReuse() throws Exception {
        IterationID iterationId = new IterationID();

        IterationRecordTypeInfo typeInfo =
                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO);
        TypeSerializer serializer = typeInfo.createSerializer(new ExecutionConfig());

        OneInputStreamOperatorTestHarness<IterationRecord<?>, Void> testHarness =
                new OneInputStreamOperatorTestHarness<>(
                        new TailOperator(iterationId, 0), serializer);
        testHarness.getExecutionConfig().enableObjectReuse();
        testHarness.open();
        SpillableFeedbackChannel channel =
                initializeFeedbackChannel(testHarness.getOperator(), iterationId, 0, 0, 0);

        IterationRecord<Integer> reuse = IterationRecord.newRecord(1, 1);
        testHarness.processElement(reuse, 2);

        reuse.setValue(2);
        testHarness.processElement(reuse, 3);

        reuse.setType(IterationRecord.Type.EPOCH_WATERMARK);
        reuse.setEpoch(2);
        reuse.setSender("sender1");
        testHarness.processElement(reuse, 4);

        List<StreamRecord<IterationRecord<?>>> iterationRecords = getFeedbackRecords(channel);
        assertEquals(
                Arrays.asList(
                        new StreamRecord<>(IterationRecord.newRecord(1, 2), 2),
                        new StreamRecord<>(IterationRecord.newRecord(2, 2), 3),
                        new StreamRecord<>(IterationRecord.newEpochWatermark(3, "sender1"), 4)),
                iterationRecords);
    }

    @Test
    public void testSpillFeedbackToDisk() throws Exception {
        IterationID iterationId = new IterationID();

        IterationRecordTypeInfo typeInfo =
                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO);
        TypeSerializer serializer = typeInfo.createSerializer(new ExecutionConfig());
        OneInputStreamOperatorTestHarness<IterationRecord<?>, Void> testHarness =
                new OneInputStreamOperatorTestHarness<>(
                        new TailOperator(iterationId, 0), serializer);
        testHarness.open();
        initializeFeedbackChannel(testHarness.getOperator(), iterationId, 0, 0, 0);

        File spillPath =
                new File(
                        testHarness
                                .getOperator()
                                .getContainingTask()
                                .getEnvironment()
                                .getIOManager()
                                .getSpillingDirectoriesPaths()[0]);
        assertEquals(0, spillPath.listFiles().length);

        for (int i = 0; i < 10; i++) {
            testHarness.processElement(IterationRecord.newRecord(i, 1), i);
        }
        assertEquals(0, spillPath.listFiles().length);

        for (int i = 0; i < (1 << 16); i++) {
            testHarness.processElement(IterationRecord.newRecord(i, 1), i);
        }
        assertEquals(1, spillPath.listFiles().length);
    }

    static List<StreamRecord<IterationRecord<?>>> getFeedbackRecords(
            SpillableFeedbackChannel<StreamRecord<IterationRecord<?>>> feedbackChannel) {
        List<StreamRecord<IterationRecord<?>>> iterationRecords = new ArrayList<>();
        OperatorUtils.registerFeedbackConsumer(
                feedbackChannel, iterationRecords::add, new DirectScheduledExecutorService());
        return iterationRecords;
    }

    static SpillableFeedbackChannel<StreamRecord<IterationRecord<?>>> initializeFeedbackChannel(
            AbstractStreamOperator operator,
            IterationID iterationId,
            int feedbackIndex,
            int subtaskIndex,
            int attemptNumber)
            throws MemoryAllocationException {
        return SpillableFeedbackChannelBroker.get()
                .getChannel(
                        OperatorUtils.<StreamRecord<IterationRecord<?>>>createFeedbackKey(
                                        iterationId, feedbackIndex)
                                .withSubTaskIndex(subtaskIndex, attemptNumber),
                        channel -> OperatorUtils.initializeFeedbackChannel(channel, operator));
    }
}
