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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.OperatorStateStore;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.core.fs.CloseableRegistry;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.AbstractWrapperOperator;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.streaming.api.operators.KeyContext;
import org.apache.flink.streaming.api.operators.OperatorSnapshotFutures;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactoryUtil;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.StreamOperatorStateContext;
import org.apache.flink.streaming.api.operators.StreamTaskStateInitializer;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.ProcessingTimeService;
import org.apache.flink.streaming.runtime.tasks.StreamTask;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import java.io.IOException;
import java.util.Collections;

import static org.apache.flink.iteration.operator.OperatorUtils.processOperatorOrUdfIfSatisfy;
import static org.apache.flink.util.Preconditions.checkState;

/** The base class for the all-round wrapper operators. */
public abstract class AbstractAllRoundWrapperOperator<T, S extends StreamOperator<T>>
        extends AbstractWrapperOperator<T> {

    protected final S wrappedOperator;

    // --------------- state ---------------------------
    private int latestEpochWatermark = -1;

    private ListState<Integer> parallelismState;

    private ListState<Integer> latestEpochWatermarkState;

    @SuppressWarnings({"unchecked", "rawtypes"})
    public AbstractAllRoundWrapperOperator(
            StreamOperatorParameters<IterationRecord<T>> parameters,
            StreamOperatorFactory<T> operatorFactory) {
        super(parameters, operatorFactory);

        this.wrappedOperator =
                (S)
                        StreamOperatorFactoryUtil.<T, S>createOperator(
                                        operatorFactory,
                                        (StreamTask) parameters.getContainingTask(),
                                        OperatorUtils.createWrappedOperatorConfig(
                                                parameters.getStreamConfig(),
                                                containingTask.getUserCodeClassLoader()),
                                        proxyOutput,
                                        parameters.getOperatorEventDispatcher())
                                .f0;

        processOperatorOrUdfIfSatisfy(
                wrappedOperator,
                EpochAware.class,
                epochWatermarkAware ->
                        epochWatermarkAware.setEpochSupplier(epochWatermarkSupplier));
    }

    @Override
    public void onEpochWatermarkIncrement(int epochWatermark) throws IOException {
        if (epochWatermark > latestEpochWatermark) {
            latestEpochWatermark = epochWatermark;

            setIterationContextRound(epochWatermark);
            processOperatorOrUdfIfSatisfy(
                    wrappedOperator,
                    IterationListener.class,
                    listener -> notifyEpochWatermarkIncrement(listener, epochWatermark));
            clearIterationContextRound();
        }

        // Always broadcasts the events.
        super.onEpochWatermarkIncrement(epochWatermark);
    }

    @Override
    public void initializeState(StreamTaskStateInitializer streamTaskStateManager)
            throws Exception {
        RecordingStreamTaskStateInitializer recordingStreamTaskStateInitializer =
                new RecordingStreamTaskStateInitializer(streamTaskStateManager);
        wrappedOperator.initializeState(recordingStreamTaskStateInitializer);
        checkState(recordingStreamTaskStateInitializer.lastCreated != null);

        OperatorStateStore operatorStateStore =
                recordingStreamTaskStateInitializer.lastCreated.operatorStateBackend();

        parallelismState =
                operatorStateStore.getUnionListState(
                        new ListStateDescriptor<>("parallelism", IntSerializer.INSTANCE));
        OperatorStateUtils.getUniqueElement(parallelismState, "parallelism")
                .ifPresent(
                        oldParallelism ->
                                checkState(
                                        oldParallelism
                                                == containingTask
                                                        .getEnvironment()
                                                        .getTaskInfo()
                                                        .getNumberOfParallelSubtasks(),
                                        "The all-round wrapper operator is recovered with parallelism changed from "
                                                + oldParallelism
                                                + " to "
                                                + containingTask
                                                        .getEnvironment()
                                                        .getTaskInfo()
                                                        .getNumberOfParallelSubtasks()));

        latestEpochWatermarkState =
                operatorStateStore.getListState(
                        new ListStateDescriptor<>("latestEpoch", IntSerializer.INSTANCE));
        OperatorStateUtils.getUniqueElement(latestEpochWatermarkState, "latestEpoch")
                .ifPresent(
                        oldLatestEpochWatermark -> latestEpochWatermark = oldLatestEpochWatermark);
    }

    @Override
    public OperatorSnapshotFutures snapshotState(
            long checkpointId,
            long timestamp,
            CheckpointOptions checkpointOptions,
            CheckpointStreamFactory storageLocation)
            throws Exception {

        // Always clear the union list state before set value.
        parallelismState.clear();
        if (containingTask.getEnvironment().getTaskInfo().getIndexOfThisSubtask() == 0) {
            parallelismState.update(
                    Collections.singletonList(
                            containingTask
                                    .getEnvironment()
                                    .getTaskInfo()
                                    .getNumberOfParallelSubtasks()));
        }
        latestEpochWatermarkState.update(Collections.singletonList(latestEpochWatermark));

        return wrappedOperator.snapshotState(
                checkpointId, timestamp, checkpointOptions, storageLocation);
    }

    @Override
    public void open() throws Exception {
        wrappedOperator.open();
    }

    @Override
    public void finish() throws Exception {
        setIterationContextRound(Integer.MAX_VALUE);
        wrappedOperator.finish();
        clearIterationContextRound();
    }

    @Override
    public void close() throws Exception {
        setIterationContextRound(Integer.MAX_VALUE);
        wrappedOperator.close();
        clearIterationContextRound();
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        wrappedOperator.prepareSnapshotPreBarrier(checkpointId);
    }

    @Override
    public void setKeyContextElement1(StreamRecord<?> record) throws Exception {
        wrappedOperator.setKeyContextElement1(record);
    }

    @Override
    public void setKeyContextElement2(StreamRecord<?> record) throws Exception {
        wrappedOperator.setKeyContextElement2(record);
    }

    @Override
    public OperatorMetricGroup getMetricGroup() {
        return wrappedOperator.getMetricGroup();
    }

    @Override
    public OperatorID getOperatorID() {
        return wrappedOperator.getOperatorID();
    }

    @Override
    public void notifyCheckpointComplete(long checkpointId) throws Exception {
        wrappedOperator.notifyCheckpointComplete(checkpointId);
    }

    @Override
    public void notifyCheckpointAborted(long checkpointId) throws Exception {
        wrappedOperator.notifyCheckpointAborted(checkpointId);
    }

    @Override
    public void setCurrentKey(Object key) {
        wrappedOperator.setCurrentKey(key);
    }

    @Override
    public Object getCurrentKey() {
        return wrappedOperator.getCurrentKey();
    }

    @VisibleForTesting
    int getLatestEpochWatermark() {
        return latestEpochWatermark;
    }

    private static class RecordingStreamTaskStateInitializer implements StreamTaskStateInitializer {

        private final StreamTaskStateInitializer wrapped;

        StreamOperatorStateContext lastCreated;

        public RecordingStreamTaskStateInitializer(StreamTaskStateInitializer wrapped) {
            this.wrapped = wrapped;
        }

        @Override
        public StreamOperatorStateContext streamOperatorStateContext(
                @Nonnull OperatorID operatorID,
                @Nonnull String s,
                @Nonnull ProcessingTimeService processingTimeService,
                @Nonnull KeyContext keyContext,
                @Nullable TypeSerializer<?> typeSerializer,
                @Nonnull CloseableRegistry closeableRegistry,
                @Nonnull MetricGroup metricGroup,
                double v,
                boolean b)
                throws Exception {
            lastCreated =
                    wrapped.streamOperatorStateContext(
                            operatorID,
                            s,
                            processingTimeService,
                            keyContext,
                            typeSerializer,
                            closeableRegistry,
                            metricGroup,
                            v,
                            b);
            return lastCreated;
        }
    }
}
