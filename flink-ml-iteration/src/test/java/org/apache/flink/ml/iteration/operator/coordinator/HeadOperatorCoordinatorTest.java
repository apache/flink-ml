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
import org.apache.flink.runtime.operators.coordination.EventReceivingTasks;
import org.apache.flink.runtime.operators.coordination.MockOperatorCoordinatorContext;
import org.apache.flink.runtime.operators.coordination.OperatorEvent;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;

import static org.junit.Assert.assertEquals;

/** Tests the behavior of {@link HeadOperatorCoordinator}. */
public class HeadOperatorCoordinatorTest {

    @Test(timeout = 60000L)
    public void testForwardEvents() throws InterruptedException {
        IterationID iterationId = new IterationID();
        List<OperatorID> operatorIds = Arrays.asList(new OperatorID(), new OperatorID());
        List<Integer> parallelisms = Arrays.asList(2, 3);
        List<EventReceivingTasks> receivingTasks =
                Arrays.asList(
                        EventReceivingTasks.createForRunningTasks(),
                        EventReceivingTasks.createForRunningTasks());
        List<HeadOperatorCoordinator> coordinators = new ArrayList<>();

        int totalParallelism = parallelisms.stream().mapToInt(i -> i).sum();

        for (int i = 0; i < operatorIds.size(); ++i) {
            HeadOperatorCoordinator coordinator =
                    createCoordinator(iterationId, parallelisms.get(i), totalParallelism);
            setAllSubtasksReady(coordinator, receivingTasks.get(i), parallelisms.get(i));
            coordinators.add(coordinator);
        }

        receiveEvent(
                coordinators,
                parallelisms,
                (i, j) -> Collections.singletonList(new SubtaskAlignedEvent(2, j, false)));
        checkSentEvent(1, new GloballyAlignedEvent(2, false), receivingTasks, parallelisms);

        receiveEvent(
                coordinators,
                parallelisms,
                (i, j) -> Collections.singletonList(new SubtaskAlignedEvent(3, 0, false)));
        checkSentEvent(2, new GloballyAlignedEvent(3, true), receivingTasks, parallelisms);
    }

    private HeadOperatorCoordinator createCoordinator(
            IterationID iterationId, int parallelism, int totalHeadParallelism) {
        MockOperatorCoordinatorContext context =
                new MockOperatorCoordinatorContext(new OperatorID(), parallelism);
        return (HeadOperatorCoordinator)
                new HeadOperatorCoordinator.HeadOperatorCoordinatorProvider(
                                new OperatorID(), iterationId, totalHeadParallelism)
                        .create(context);
    }

    private void setAllSubtasksReady(
            HeadOperatorCoordinator coordinator,
            EventReceivingTasks receivingTasks,
            int parallelism) {
        for (int i = 0; i < parallelism; i++) {
            coordinator.subtaskReady(i, receivingTasks.createGatewayForSubtask(i));
        }
    }

    private void receiveEvent(
            List<HeadOperatorCoordinator> coordinators,
            List<Integer> parallelisms,
            BiFunction<Integer, Integer, List<OperatorEvent>> eventFactory) {
        for (int i = 0; i < coordinators.size(); ++i) {
            for (int j = 0; j < parallelisms.get(i); ++j) {
                List<OperatorEvent> events = eventFactory.apply(i, j);
                for (OperatorEvent event : events) {
                    coordinators.get(i).handleEventFromOperator(j, event);
                }
            }
        }
    }

    private void checkSentEvent(
            int expectedNumEvents,
            GloballyAlignedEvent expectedLastEvent,
            List<EventReceivingTasks> receivingTasks,
            List<Integer> parallelisms)
            throws InterruptedException {
        for (int i = 0; i < parallelisms.size(); ++i) {
            for (int j = 0; j < parallelisms.get(i); ++j) {
                while (true) {
                    List<OperatorEvent> events = receivingTasks.get(i).getSentEventsForSubtask(j);
                    if (events.size() < expectedNumEvents) {
                        Thread.sleep(50);
                        continue;
                    }

                    assertEquals(expectedLastEvent, events.get(events.size() - 1));
                    break;
                }
            }
        }
    }
}
