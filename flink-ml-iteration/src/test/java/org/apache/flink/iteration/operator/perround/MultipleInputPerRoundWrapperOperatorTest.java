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

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.iteration.operator.WrapperOperatorFactory;
import org.apache.flink.iteration.operator.allround.LifeCycle;
import org.apache.flink.iteration.operator.allround.OneInputAllRoundWrapperOperator;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.runtime.checkpoint.CheckpointMetaData;
import org.apache.flink.runtime.checkpoint.CheckpointMetricsBuilder;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.checkpoint.CheckpointType;
import org.apache.flink.runtime.io.network.api.EndOfData;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.CheckpointStorageLocationReference;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractInput;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorV2;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.MultipleInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/** Tests the {@link OneInputAllRoundWrapperOperator}. */
public class MultipleInputPerRoundWrapperOperatorTest {

    private static List<LifeCycle> lifeCycles = new ArrayList<>();

    @Test
    public void testProcessElementsAndEpochWatermarks() throws Exception {
        StreamOperatorFactory<IterationRecord<Integer>> wrapperFactory =
                new WrapperOperatorFactory<>(
                        new LifeCycleTrackingMultiInputStreamOperatorFactory(),
                        new PerRoundOperatorWrapper<>());
        OperatorID operatorId = new OperatorID();

        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                MultipleInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .setupOutputForSingletonOperatorChain(wrapperFactory, operatorId)
                        .build()) {
            harness.processElement(new StreamRecord<>(IterationRecord.newRecord(5, 1), 2), 0);
            harness.processElement(new StreamRecord<>(IterationRecord.newRecord(6, 2), 3), 2);

            // Check the output
            assertEquals(
                    Arrays.asList(
                            new StreamRecord<>(IterationRecord.newRecord(5, 1), 2),
                            new StreamRecord<>(IterationRecord.newRecord(6, 2), 3)),
                    new ArrayList<>(harness.getOutput()));

            // Check the other lifecycles.
            harness.getStreamTask()
                    .triggerCheckpointOnBarrier(
                            new CheckpointMetaData(5, 2),
                            CheckpointOptions.alignedNoTimeout(
                                    CheckpointType.CHECKPOINT,
                                    CheckpointStorageLocationReference.getDefault()),
                            new CheckpointMetricsBuilder()
                                    .setAlignmentDurationNanos(0)
                                    .setBytesProcessedDuringAlignment(0));
            harness.processAll();

            harness.getStreamTask().notifyCheckpointCompleteAsync(5);
            harness.processAll();

            harness.getStreamTask().notifyCheckpointAbortAsync(6, 5);
            harness.processAll();

            harness.getOutput().clear();
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(1, "only-one-0")), 0);
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(1, "only-one-1")), 1);
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(1, "only-one-2")), 2);
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(2, "only-one-0")), 0);
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(2, "only-one-1")), 1);
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(2, "only-one-2")), 2);

            // Check the output
            assertEquals(
                    Arrays.asList(
                            new StreamRecord<>(
                                    IterationRecord.newEpochWatermark(
                                            1, OperatorUtils.getUniqueSenderId(operatorId, 0))),
                            new StreamRecord<>(
                                    IterationRecord.newEpochWatermark(
                                            2, OperatorUtils.getUniqueSenderId(operatorId, 0)))),
                    new ArrayList<>(harness.getOutput()));

            harness.processEvent(EndOfData.INSTANCE, 0);
            harness.processEvent(EndOfData.INSTANCE, 1);
            harness.processEvent(EndOfData.INSTANCE, 2);
            harness.endInput();
            harness.finishProcessing();

            assertEquals(
                    Arrays.asList(
                            /* First wrapped operator */
                            LifeCycle.INITIALIZE_STATE,
                            LifeCycle.OPEN,
                            LifeCycle.PROCESS_ELEMENT,
                            /* second wrapped operator */
                            LifeCycle.INITIALIZE_STATE,
                            LifeCycle.OPEN,
                            LifeCycle.PROCESS_ELEMENT,
                            /* states */
                            LifeCycle.PREPARE_SNAPSHOT_PRE_BARRIER,
                            LifeCycle.PREPARE_SNAPSHOT_PRE_BARRIER,
                            LifeCycle.SNAPSHOT_STATE,
                            LifeCycle.SNAPSHOT_STATE,
                            LifeCycle.NOTIFY_CHECKPOINT_COMPLETE,
                            LifeCycle.NOTIFY_CHECKPOINT_COMPLETE,
                            LifeCycle.NOTIFY_CHECKPOINT_ABORT,
                            LifeCycle.NOTIFY_CHECKPOINT_ABORT,
                            LifeCycle.FINISH,
                            LifeCycle.CLOSE,
                            LifeCycle.FINISH,
                            LifeCycle.CLOSE),
                    lifeCycles);
        }
    }

    private static class LifeCycleTrackingMultiInputStreamOperator
            extends AbstractStreamOperatorV2<Integer>
            implements MultipleInputStreamOperator<Integer> {

        private final int numberOfInputs;

        public LifeCycleTrackingMultiInputStreamOperator(
                StreamOperatorParameters<Integer> parameters, int numberOfInputs) {
            super(parameters, numberOfInputs);
            this.numberOfInputs = numberOfInputs;
        }

        @Override
        public List<Input> getInputs() {
            List<Input> inputs = new ArrayList<>();
            for (int i = 0; i < numberOfInputs; ++i) {
                inputs.add(
                        new AbstractInput(this, i + 1) {
                            @Override
                            public void processElement(StreamRecord element) throws Exception {
                                output.collect(element);
                                lifeCycles.add(LifeCycle.PROCESS_ELEMENT);
                            }
                        });
            }

            return inputs;
        }

        @Override
        public void open() throws Exception {
            super.open();
            lifeCycles.add(LifeCycle.OPEN);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            lifeCycles.add(LifeCycle.INITIALIZE_STATE);
        }

        @Override
        public void finish() throws Exception {
            super.finish();
            lifeCycles.add(LifeCycle.FINISH);
        }

        @Override
        public void close() throws Exception {
            super.close();
            lifeCycles.add(LifeCycle.CLOSE);
        }

        @Override
        public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
            super.prepareSnapshotPreBarrier(checkpointId);
            lifeCycles.add(LifeCycle.PREPARE_SNAPSHOT_PRE_BARRIER);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            lifeCycles.add(LifeCycle.SNAPSHOT_STATE);
        }

        @Override
        public void notifyCheckpointComplete(long checkpointId) throws Exception {
            super.notifyCheckpointComplete(checkpointId);
            lifeCycles.add(LifeCycle.NOTIFY_CHECKPOINT_COMPLETE);
        }

        @Override
        public void notifyCheckpointAborted(long checkpointId) throws Exception {
            super.notifyCheckpointAborted(checkpointId);
            lifeCycles.add(LifeCycle.NOTIFY_CHECKPOINT_ABORT);
        }
    }

    /** Life-cycle tracking stream operator factory. */
    public static class LifeCycleTrackingMultiInputStreamOperatorFactory
            extends AbstractStreamOperatorFactory<Integer> {

        @Override
        public <T extends StreamOperator<Integer>> T createStreamOperator(
                StreamOperatorParameters<Integer> parameters) {
            return (T) new LifeCycleTrackingMultiInputStreamOperator(parameters, 3);
        }

        @Override
        public Class<? extends StreamOperator> getStreamOperatorClass(ClassLoader classLoader) {
            return LifeCycleTrackingMultiInputStreamOperator.class;
        }
    }
}
