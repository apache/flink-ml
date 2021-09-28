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

package org.apache.flink.ml.iteration.operator.coordinator;

import org.apache.flink.ml.iteration.IterationID;
import org.apache.flink.ml.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.ml.iteration.operator.event.SubtaskAlignedEvent;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.operators.coordination.MockOperatorCoordinatorContext;
import org.apache.flink.runtime.testutils.DirectScheduledExecutorService;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/** Tests the {@link SharedProgressAligner}. */
public class SharedProgressAlignerTest {

    @Test
    public void testCreateAndGet() {
        IterationID iterationId = new IterationID();
        int firstHeadParallelism = 2;
        int secondHeadParallelism = 3;

        SharedProgressAligner firstAligner =
                SharedProgressAligner.getOrCreate(
                        iterationId,
                        firstHeadParallelism + secondHeadParallelism,
                        new MockOperatorCoordinatorContext(new OperatorID(), firstHeadParallelism),
                        DirectScheduledExecutorService::new);
        SharedProgressAligner secondAligner =
                SharedProgressAligner.getOrCreate(
                        iterationId,
                        firstHeadParallelism + secondHeadParallelism,
                        new MockOperatorCoordinatorContext(new OperatorID(), secondHeadParallelism),
                        DirectScheduledExecutorService::new);
        assertSame(firstAligner, secondAligner);
    }

    @Test
    public void testRegisterAndUnregisterConsumers() {
        IterationID iterationId = new IterationID();
        List<OperatorID> operatorIds = Arrays.asList(new OperatorID(), new OperatorID());
        List<Consumer<GloballyAlignedEvent>> consumers =
                Arrays.asList(new RecordingConsumer(), new RecordingConsumer());
        SharedProgressAligner aligner =
                initializeAligner(iterationId, operatorIds, Arrays.asList(2, 3), consumers);

        assertEquals(2, aligner.getNumberConsumers());

        aligner.unregisterConsumer(operatorIds.get(0));
        assertEquals(1, aligner.getNumberConsumers());
        assertTrue(SharedProgressAligner.getInstances().containsKey(iterationId));

        aligner.unregisterConsumer(operatorIds.get(1));
        assertEquals(0, aligner.getNumberConsumers());
        assertFalse(SharedProgressAligner.getInstances().containsKey(iterationId));
    }

    @Test
    public void testNonTerminatedAlignment() {
        IterationID iterationId = new IterationID();
        List<OperatorID> operatorIds = Arrays.asList(new OperatorID(), new OperatorID());
        List<Integer> parallelisms = Arrays.asList(2, 3);
        List<RecordingConsumer> consumers =
                Arrays.asList(new RecordingConsumer(), new RecordingConsumer());
        SharedProgressAligner aligner =
                initializeAligner(iterationId, operatorIds, parallelisms, consumers);

        for (int i = 0; i < operatorIds.size(); ++i) {
            for (int j = 0; j < parallelisms.get(i); ++j) {
                aligner.reportSubtaskProgress(
                        operatorIds.get(i), j, new SubtaskAlignedEvent(2, i + j, false));
            }
        }

        checkRecordingConsumers(
                Collections.singletonList(new GloballyAlignedEvent(2, false)), consumers);
    }

    @Test
    public void testTerminateIfNoRecords() {
        IterationID iterationId = new IterationID();
        List<OperatorID> operatorIds = Arrays.asList(new OperatorID(), new OperatorID());
        List<Integer> parallelisms = Arrays.asList(2, 3);
        List<RecordingConsumer> consumers =
                Arrays.asList(new RecordingConsumer(), new RecordingConsumer());
        SharedProgressAligner aligner =
                initializeAligner(iterationId, operatorIds, parallelisms, consumers);

        for (int i = 0; i < operatorIds.size(); ++i) {
            for (int j = 0; j < parallelisms.get(i); ++j) {
                aligner.reportSubtaskProgress(
                        operatorIds.get(i), j, new SubtaskAlignedEvent(2, 0, false));
            }
        }

        checkRecordingConsumers(
                Collections.singletonList(new GloballyAlignedEvent(2, true)), consumers);
    }

    @Test
    public void testNotTerminateForRoundZero() {
        IterationID iterationId = new IterationID();
        List<OperatorID> operatorIds = Arrays.asList(new OperatorID(), new OperatorID());
        List<Integer> parallelisms = Arrays.asList(2, 3);
        List<RecordingConsumer> consumers =
                Arrays.asList(new RecordingConsumer(), new RecordingConsumer());
        SharedProgressAligner aligner =
                initializeAligner(iterationId, operatorIds, parallelisms, consumers);

        for (int i = 0; i < operatorIds.size(); ++i) {
            for (int j = 0; j < parallelisms.get(i); ++j) {
                aligner.reportSubtaskProgress(
                        operatorIds.get(i), j, new SubtaskAlignedEvent(0, 0, false));
            }
        }

        checkRecordingConsumers(
                Collections.singletonList(new GloballyAlignedEvent(0, false)), consumers);
    }

    @Test
    public void testTerminateIfCriteriaStreamNoRecords() {
        IterationID iterationId = new IterationID();
        List<OperatorID> operatorIds = Arrays.asList(new OperatorID(), new OperatorID());
        List<Integer> parallelisms = Arrays.asList(2, 3);
        List<RecordingConsumer> consumers =
                Arrays.asList(new RecordingConsumer(), new RecordingConsumer());
        SharedProgressAligner aligner =
                initializeAligner(iterationId, operatorIds, parallelisms, consumers);

        for (int i = 0; i < operatorIds.size(); ++i) {
            for (int j = 0; j < parallelisms.get(i); ++j) {
                // Operator 0 is the criteria stream
                aligner.reportSubtaskProgress(
                        operatorIds.get(i), j, new SubtaskAlignedEvent(2, i == 0 ? 0 : j, i == 0));
            }
        }

        checkRecordingConsumers(
                Collections.singletonList(new GloballyAlignedEvent(2, true)), consumers);
    }

    private SharedProgressAligner initializeAligner(
            IterationID iterationId,
            List<OperatorID> operatorIds,
            List<Integer> parallelisms,
            List<? extends Consumer<GloballyAlignedEvent>> consumers) {

        SharedProgressAligner aligner =
                SharedProgressAligner.getOrCreate(
                        iterationId,
                        parallelisms.stream().mapToInt(i -> i).sum(),
                        new MockOperatorCoordinatorContext(operatorIds.get(0), parallelisms.get(0)),
                        DirectScheduledExecutorService::new);

        for (int i = 0; i < consumers.size(); ++i) {
            aligner.registerAlignedConsumer(operatorIds.get(i), consumers.get(i));
        }

        return aligner;
    }

    private void checkRecordingConsumers(
            List<GloballyAlignedEvent> expectedGloballyAlignedEvents,
            List<RecordingConsumer> consumers) {
        for (RecordingConsumer consumer : consumers) {
            assertEquals(expectedGloballyAlignedEvents, consumer.globallyAlignedEvents);
        }
    }

    private static class RecordingConsumer implements Consumer<GloballyAlignedEvent> {

        final List<GloballyAlignedEvent> globallyAlignedEvents = new ArrayList<>();

        @Override
        public void accept(GloballyAlignedEvent globallyAlignedEvent) {
            globallyAlignedEvents.add(globallyAlignedEvent);
        }
    }
}
