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

import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;
import org.apache.flink.streaming.util.TestHarnessUtil;

import org.junit.Test;

import java.util.concurrent.ConcurrentLinkedQueue;

/** Tests the {@link OutputOperator}. */
public class OutputOperatorTest {

    @Test
    public void testUnwrapUserRecordsAndDropEvents() throws Exception {

        OneInputStreamOperatorTestHarness<IterationRecord<Integer>, Integer> testHarness =
                new OneInputStreamOperatorTestHarness<>(new OutputOperator<>());
        testHarness.open();

        testHarness.processElement(IterationRecord.newRecord(1, 0), 1);
        testHarness.processElement(IterationRecord.newRecord(2, 3), 2);
        testHarness.processElement(IterationRecord.newRecord(2, 4), 3);
        testHarness.processElement(IterationRecord.newEpochWatermark(2, "sender1"), 4);
        testHarness.processElement(IterationRecord.newBarrier(5), 4);

        ConcurrentLinkedQueue<Object> expectedOutput = new ConcurrentLinkedQueue<>();
        expectedOutput.add(new StreamRecord<>(1, 1));
        expectedOutput.add(new StreamRecord<>(2, 2));
        expectedOutput.add(new StreamRecord<>(2, 3));

        TestHarnessUtil.assertOutputEquals(
                "Output was not correct", expectedOutput, testHarness.getOutput());
    }
}
