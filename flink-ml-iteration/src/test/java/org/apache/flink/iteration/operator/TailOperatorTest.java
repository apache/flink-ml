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

import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.runtime.testutils.DirectScheduledExecutorService;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannel;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannelBroker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/** Tests the {@link TailOperator}. */
public class TailOperatorTest {

    @Test
    public void testIncrementRoundWithoutObjectReuse() throws Exception {
        IterationID iterationId = new IterationID();

        OneInputStreamOperatorTestHarness<IterationRecord<?>, Void> testHarness =
                new OneInputStreamOperatorTestHarness<>(new TailOperator(iterationId, 0));
        testHarness.open();

        testHarness.processElement(IterationRecord.newRecord(1, 1), 2);
        testHarness.processElement(IterationRecord.newRecord(2, 1), 3);
        testHarness.processElement(IterationRecord.newEpochWatermark(2, "sender1"), 4);

        List<StreamRecord<IterationRecord<?>>> iterationRecords =
                getFeedbackRecords(iterationId, 0, 0, 0);
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

        OneInputStreamOperatorTestHarness<IterationRecord<?>, Void> testHarness =
                new OneInputStreamOperatorTestHarness<>(new TailOperator(iterationId, 0));
        testHarness.getExecutionConfig().enableObjectReuse();
        testHarness.open();

        IterationRecord<Integer> reuse = IterationRecord.newRecord(1, 1);
        testHarness.processElement(reuse, 2);

        reuse.setValue(2);
        testHarness.processElement(reuse, 3);

        reuse.setType(IterationRecord.Type.EPOCH_WATERMARK);
        reuse.setEpoch(2);
        reuse.setSender("sender1");
        testHarness.processElement(reuse, 4);

        List<StreamRecord<IterationRecord<?>>> iterationRecords =
                getFeedbackRecords(iterationId, 0, 0, 0);
        assertEquals(
                Arrays.asList(
                        new StreamRecord<>(IterationRecord.newRecord(1, 2), 2),
                        new StreamRecord<>(IterationRecord.newRecord(2, 2), 3),
                        new StreamRecord<>(IterationRecord.newEpochWatermark(3, "sender1"), 4)),
                iterationRecords);
    }

    static List<StreamRecord<IterationRecord<?>>> getFeedbackRecords(
            IterationID iterationId, int feedbackIndex, int subtaskIndex, int attemptNumber) {
        FeedbackChannel<StreamRecord<IterationRecord<?>>> feedbackChannel =
                FeedbackChannelBroker.get()
                        .getChannel(
                                OperatorUtils.<StreamRecord<IterationRecord<?>>>createFeedbackKey(
                                                iterationId, feedbackIndex)
                                        .withSubTaskIndex(subtaskIndex, attemptNumber));
        List<StreamRecord<IterationRecord<?>>> iterationRecords = new ArrayList<>();
        OperatorUtils.registerFeedbackConsumer(
                feedbackChannel, iterationRecords::add, new DirectScheduledExecutorService());
        return iterationRecords;
    }
}
