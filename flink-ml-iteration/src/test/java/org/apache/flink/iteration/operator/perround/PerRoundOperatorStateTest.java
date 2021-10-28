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

package org.apache.flink.iteration.operator.perround;

import org.apache.flink.api.common.state.BroadcastState;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.java.typeutils.EnumTypeInfo;
import org.apache.flink.contrib.streaming.state.EmbeddedRocksDBStateBackend;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.WrapperOperatorFactory;
import org.apache.flink.iteration.proxy.ProxyKeySelector;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.runtime.state.StateBackend;
import org.apache.flink.runtime.state.hashmap.HashMapStateBackend;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.operators.KeyedProcessOperator;
import org.apache.flink.streaming.api.operators.ProcessOperator;
import org.apache.flink.streaming.api.operators.SimpleOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.OneInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;
import org.apache.flink.util.Collector;

import org.junit.Test;

import javax.annotation.Nullable;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/** Tests the state isolation and cleanup for the per-round operators. */
public class PerRoundOperatorStateTest {

    @Test
    public void testStateIsolationWithoutKeyedStateBackend() throws Exception {
        testStateIsolation(null);
    }

    @Test
    public void testStateIsolationWithHashMapKeyedStateBackend() throws Exception {
        testStateIsolation(new HashMapStateBackend());
    }

    @Test
    public void testStateIsolationWithRocksDBKeyedStateBackend() throws Exception {
        testStateIsolation(new EmbeddedRocksDBStateBackend());
    }

    private void testStateIsolation(@Nullable StateBackend stateBackend) throws Exception {
        StreamOperatorFactory<IterationRecord<Integer>> wrapperFactory =
                new WrapperOperatorFactory<>(
                        SimpleOperatorFactory.of(
                                stateBackend == null
                                        ? new ProcessOperator<>(new StatefulProcessFunction())
                                        : new KeyedProcessOperator<>(
                                                new KeyedStatefulProcessFunction())),
                        new PerRoundOperatorWrapper<>());
        OperatorID operatorId = new OperatorID();

        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                OneInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .modifyStreamConfig(
                                streamConfig -> {
                                    if (stateBackend != null) {
                                        streamConfig.setStateBackend(stateBackend);
                                        streamConfig.setManagedMemoryFractionOperatorOfUseCase(
                                                ManagedMemoryUseCase.STATE_BACKEND, 0.2);
                                        streamConfig.setStateKeySerializer(IntSerializer.INSTANCE);
                                    }
                                })
                        .addInput(
                                new IterationRecordTypeInfo<>(new EnumTypeInfo<>(ActionType.class)),
                                1,
                                stateBackend == null ? null : new ProxyKeySelector<>(x -> 10))
                        .setupOutputForSingletonOperatorChain(wrapperFactory, operatorId)
                        .build()) {

            // Set round 0
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newRecord(ActionType.SET, 0)), 0);
            testGetRound(harness, Arrays.asList(10, 10, stateBackend == null ? -1 : 10), 0);
            testGetRound(harness, Arrays.asList(-1, -1, -1), 1);

            // Set round 1
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newRecord(ActionType.SET, 1)), 0);
            testGetRound(harness, Arrays.asList(10, 10, stateBackend == null ? -1 : 10), 1);

            // Clear round 0. Although after round 0 we should not receive records for round 0 in
            // realistic, we use this method to check the current value of states.
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(0, "sender")), 0);
            testGetRound(harness, Arrays.asList(-1, -1, -1), 0);
            testGetRound(harness, Arrays.asList(10, 10, stateBackend == null ? -1 : 10), 1);

            // Clear round 1
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(1, "sender")), 0);
            testGetRound(harness, Arrays.asList(-1, -1, -1), 0);
            testGetRound(harness, Arrays.asList(-1, -1, -1), 1);
        }
    }

    private void testGetRound(
            StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness,
            List<Integer> expectedValues,
            int round)
            throws Exception {
        harness.getOutput().clear();
        harness.processElement(
                new StreamRecord<>(IterationRecord.newRecord(ActionType.GET, round)), 0);

        assertEquals(
                expectedValues.stream()
                        .map(i -> IterationRecord.newRecord(i, round))
                        .collect(Collectors.toList()),
                harness.getOutput().stream()
                        .map(r -> ((StreamRecord<?>) r).getValue())
                        .collect(Collectors.toList()));
    }

    enum ActionType {
        SET,
        GET
    }

    private static class StatefulProcessFunction extends ProcessFunction<ActionType, Integer>
            implements CheckpointedFunction {

        private transient BroadcastState<Integer, Integer> broadcastState;

        private transient ListState<Integer> operatorState;

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            this.operatorState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("opState", IntSerializer.INSTANCE));
            this.broadcastState =
                    context.getOperatorStateStore()
                            .getBroadcastState(
                                    new MapStateDescriptor<>(
                                            "broadState",
                                            IntSerializer.INSTANCE,
                                            IntSerializer.INSTANCE));
        }

        @Override
        public void snapshotState(FunctionSnapshotContext context) throws Exception {}

        @Override
        public void processElement(ActionType value, Context ctx, Collector<Integer> out)
                throws Exception {
            switch (value) {
                case SET:
                    operatorState.add(10);
                    broadcastState.put(10, 10);
                    break;
                case GET:
                    out.collect(
                            operatorState.get().iterator().hasNext()
                                    ? operatorState.get().iterator().next()
                                    : -1);
                    out.collect(mapNullToMinusOne(broadcastState.get(10)));
                    // To keep the same amount of outputs with the keyed one.
                    out.collect(mapNullToMinusOne(null));
                    break;
            }
        }
    }

    private static class KeyedStatefulProcessFunction
            extends KeyedProcessFunction<Integer, ActionType, Integer>
            implements CheckpointedFunction {

        private transient BroadcastState<Integer, Integer> broadcastState;

        private transient ListState<Integer> operatorState;

        private transient ValueState<Integer> keyedState;

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            this.operatorState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("opState", IntSerializer.INSTANCE));
            this.broadcastState =
                    context.getOperatorStateStore()
                            .getBroadcastState(
                                    new MapStateDescriptor<>(
                                            "broadState",
                                            IntSerializer.INSTANCE,
                                            IntSerializer.INSTANCE));
            this.keyedState =
                    context.getKeyedStateStore()
                            .getState(
                                    new ValueStateDescriptor<>(
                                            "keyedState", IntSerializer.INSTANCE));
        }

        @Override
        public void snapshotState(FunctionSnapshotContext context) throws Exception {}

        @Override
        public void processElement(ActionType value, Context ctx, Collector<Integer> out)
                throws Exception {
            switch (value) {
                case SET:
                    operatorState.add(10);
                    broadcastState.put(10, 10);
                    keyedState.update(10);
                    break;
                case GET:
                    out.collect(
                            operatorState.get().iterator().hasNext()
                                    ? operatorState.get().iterator().next()
                                    : -1);
                    out.collect(mapNullToMinusOne(broadcastState.get(10)));
                    out.collect(mapNullToMinusOne(keyedState == null ? null : keyedState.value()));
                    break;
            }
        }
    }

    private static int mapNullToMinusOne(Integer value) {
        return value == null ? -1 : value;
    }
}
