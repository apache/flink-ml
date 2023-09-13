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

package org.apache.flink.iteration.operator.allround;

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.iteration.operator.OperatorWrapper;
import org.apache.flink.iteration.operator.WrapperOperatorFactory;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.runtime.checkpoint.CheckpointMetaData;
import org.apache.flink.runtime.checkpoint.CheckpointMetricsBuilder;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.checkpoint.CheckpointType;
import org.apache.flink.runtime.checkpoint.TaskStateSnapshot;
import org.apache.flink.runtime.io.network.api.EndOfData;
import org.apache.flink.runtime.io.network.api.StopMode;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.CheckpointStorageLocationReference;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.SimpleOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.OneInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;
import org.apache.flink.util.Collector;
import org.apache.flink.util.TestLogger;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/** Tests the {@link OneInputAllRoundWrapperOperator}. */
public class OneInputAllRoundWrapperOperatorTest extends TestLogger {

    private static final List<LifeCycle> LIFE_CYCLES = new ArrayList<>();

    @Test
    public void testProcessElementsAndEpochWatermarks() throws Exception {
        StreamOperatorFactory<IterationRecord<Integer>> wrapperFactory =
                new WrapperOperatorFactory<>(
                        SimpleOperatorFactory.of(new LifeCycleTrackingOneInputStreamOperator()),
                        new AllRoundOperatorWrapper<>());
        OperatorID operatorId = new OperatorID();

        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                OneInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .setupOutputForSingletonOperatorChain(wrapperFactory, operatorId)
                        .build()) {
            harness.processElement(new StreamRecord<>(IterationRecord.newRecord(5, 1), 2));
            harness.processElement(new StreamRecord<>(IterationRecord.newRecord(6, 2), 3));
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(5, "only-one")));
            harness.processElement(
                    new StreamRecord<>(
                            IterationRecord.newEpochWatermark(
                                    IterationRecord.END_EPOCH_WATERMARK, "only-one")));

            // Checks the output
            assertEquals(
                    Arrays.asList(
                            new StreamRecord<>(IterationRecord.newRecord(5, 1), 2),
                            new StreamRecord<>(IterationRecord.newRecord(6, 2), 3),
                            new StreamRecord<>(
                                    IterationRecord.newEpochWatermark(
                                            5, OperatorUtils.getUniqueSenderId(operatorId, 0))),
                            new StreamRecord<>(
                                    IterationRecord.newEpochWatermark(
                                            IterationRecord.END_EPOCH_WATERMARK,
                                            OperatorUtils.getUniqueSenderId(operatorId, 0)))),
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

            harness.processEvent(new EndOfData(StopMode.DRAIN), 0);
            harness.endInput();
            harness.finishProcessing();

            assertEquals(
                    Arrays.asList(
                            LifeCycle.SETUP,
                            LifeCycle.INITIALIZE_STATE,
                            LifeCycle.OPEN,
                            LifeCycle.PROCESS_ELEMENT,
                            LifeCycle.PROCESS_ELEMENT,
                            LifeCycle.EPOCH_WATERMARK_INCREMENTED,
                            LifeCycle.ITERATION_TERMINATION,
                            LifeCycle.PREPARE_SNAPSHOT_PRE_BARRIER,
                            LifeCycle.SNAPSHOT_STATE,
                            LifeCycle.NOTIFY_CHECKPOINT_COMPLETE,
                            LifeCycle.NOTIFY_CHECKPOINT_ABORT,
                            LifeCycle.END_INPUT,
                            LifeCycle.FINISH,
                            LifeCycle.CLOSE),
                    LIFE_CYCLES);
        }
    }

    @Test
    public void testSnapshotAndRestore() throws Exception {
        StreamOperatorFactory<IterationRecord<Integer>> wrapperFactory =
                new RecordingOperatorFactory<>(
                        SimpleOperatorFactory.of(new LifeCycleTrackingOneInputStreamOperator()),
                        new AllRoundOperatorWrapper<>());
        OperatorID operatorId = new OperatorID();

        TaskStateSnapshot taskStateSnapshot = null;
        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                OneInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .setupOutputForSingletonOperatorChain(wrapperFactory, operatorId)
                        .build()) {
            harness.getTaskStateManager().getWaitForReportLatch().reset();
            harness.processElement(
                    new StreamRecord<>(IterationRecord.newEpochWatermark(5, "fake")));
            harness.getStreamTask()
                    .triggerCheckpointAsync(
                            new CheckpointMetaData(2, 1000),
                            CheckpointOptions.alignedNoTimeout(
                                    CheckpointType.CHECKPOINT,
                                    CheckpointStorageLocationReference.getDefault()));
            harness.processAll();

            harness.getTaskStateManager().getWaitForReportLatch().await();
            taskStateSnapshot = harness.getTaskStateManager().getLastJobManagerTaskStateSnapshot();
        }

        assertNotNull(taskStateSnapshot);
        try (StreamTaskMailboxTestHarness<IterationRecord<Integer>> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                OneInputStreamTask::new,
                                new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .setTaskStateSnapshot(2, taskStateSnapshot)
                        .addInput(new IterationRecordTypeInfo<>(BasicTypeInfo.INT_TYPE_INFO))
                        .setupOutputForSingletonOperatorChain(wrapperFactory, operatorId)
                        .build()) {
            assertEquals(
                    5,
                    ((AbstractAllRoundWrapperOperator) RecordingOperatorFactory.latest)
                            .getLatestEpochWatermark());
        }
    }

    private static class RecordingOperatorFactory<OUT> extends WrapperOperatorFactory<OUT> {

        static StreamOperator<?> latest = null;

        public RecordingOperatorFactory(
                StreamOperatorFactory<OUT> operatorFactory,
                OperatorWrapper<OUT, IterationRecord<OUT>> wrapper) {
            super(operatorFactory, wrapper);
        }

        @Override
        public <T extends StreamOperator<IterationRecord<OUT>>> T createStreamOperator(
                StreamOperatorParameters<IterationRecord<OUT>> parameters) {
            latest = super.createStreamOperator(parameters);
            return (T) latest;
        }
    }

    private static class LifeCycleTrackingOneInputStreamOperator
            extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Integer, Integer>,
                    BoundedOneInput,
                    IterationListener {

        @Override
        public void setup(
                StreamTask<?, ?> containingTask,
                StreamConfig config,
                Output<StreamRecord<Integer>> output) {
            super.setup(containingTask, config, output);
            LIFE_CYCLES.add(LifeCycle.SETUP);
        }

        @Override
        public void open() throws Exception {
            super.open();
            LIFE_CYCLES.add(LifeCycle.OPEN);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            LIFE_CYCLES.add(LifeCycle.INITIALIZE_STATE);
        }

        @Override
        public void finish() throws Exception {
            super.finish();
            LIFE_CYCLES.add(LifeCycle.FINISH);
        }

        @Override
        public void close() throws Exception {
            super.close();
            LIFE_CYCLES.add(LifeCycle.CLOSE);
        }

        @Override
        public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
            super.prepareSnapshotPreBarrier(checkpointId);
            LIFE_CYCLES.add(LifeCycle.PREPARE_SNAPSHOT_PRE_BARRIER);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            LIFE_CYCLES.add(LifeCycle.SNAPSHOT_STATE);
        }

        @Override
        public void notifyCheckpointComplete(long checkpointId) throws Exception {
            super.notifyCheckpointComplete(checkpointId);
            LIFE_CYCLES.add(LifeCycle.NOTIFY_CHECKPOINT_COMPLETE);
        }

        @Override
        public void notifyCheckpointAborted(long checkpointId) throws Exception {
            super.notifyCheckpointAborted(checkpointId);
            LIFE_CYCLES.add(LifeCycle.NOTIFY_CHECKPOINT_ABORT);
        }

        @Override
        public void processElement(StreamRecord<Integer> element) throws Exception {
            output.collect(element);
            LIFE_CYCLES.add(LifeCycle.PROCESS_ELEMENT);
        }

        @Override
        public void endInput() throws Exception {
            LIFE_CYCLES.add(LifeCycle.END_INPUT);
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector collector) {
            LIFE_CYCLES.add(LifeCycle.EPOCH_WATERMARK_INCREMENTED);
        }

        @Override
        public void onIterationTerminated(Context context, Collector collector) {
            LIFE_CYCLES.add(LifeCycle.ITERATION_TERMINATION);
        }
    }
}
