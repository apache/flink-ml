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

package org.apache.flink.ml.common.broadcast.operator;

import org.apache.flink.api.common.operators.MailboxExecutor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.ml.common.broadcast.BroadcastContext;
import org.apache.flink.ml.iteration.config.IterationOptions;
import org.apache.flink.ml.iteration.datacache.nonkeyed.DataCacheSnapshot;
import org.apache.flink.ml.iteration.datacache.nonkeyed.DataCacheWriter;
import org.apache.flink.ml.iteration.datacache.nonkeyed.Segment;
import org.apache.flink.ml.iteration.proxy.state.ProxyStreamOperatorStateContext;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.execution.Environment;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.metrics.groups.InternalOperatorIOMetricGroup;
import org.apache.flink.runtime.metrics.groups.UnregisteredMetricGroups;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.runtime.state.OperatorStateCheckpointOutputStream;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StatePartitionStreamProvider;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.util.NonClosingInputStreamDecorator;
import org.apache.flink.runtime.util.NonClosingOutpusStreamDecorator;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.InternalTimeServiceManager;
import org.apache.flink.streaming.api.operators.OperatorSnapshotFutures;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactoryUtil;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.StreamOperatorStateContext;
import org.apache.flink.streaming.api.operators.StreamOperatorStateHandler;
import org.apache.flink.streaming.api.operators.StreamOperatorStateHandler.CheckpointedStreamOperator;
import org.apache.flink.streaming.api.operators.StreamTaskStateInitializer;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.streaming.runtime.tasks.mailbox.TaskMailbox;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/** Base class for the broadcast wrapper operators. */
public abstract class AbstractBroadcastWrapperOperator<T, S extends StreamOperator<T>>
        implements StreamOperator<T>,
                BoundedMultiInput,
                StreamOperatorStateHandler.CheckpointedStreamOperator {

    private static final Logger LOG =
            LoggerFactory.getLogger(AbstractBroadcastWrapperOperator.class);

    protected final StreamOperatorParameters<T> parameters;

    protected final StreamConfig streamConfig;

    protected final StreamTask<?, ?> containingTask;

    protected final Output<StreamRecord<T>> output;

    protected final StreamOperatorFactory<T> operatorFactory;

    /** Metric group for the operator. */
    protected final OperatorMetricGroup metrics;

    protected final S wrappedOperator;

    /** variables for withBroadcast operators. */
    protected final MailboxExecutor mailboxExecutor;

    protected final String[] broadcastStreamNames;

    protected final boolean[] isBlocking;

    protected final TypeInformation[] inTypes;

    protected boolean broadcastVariablesReady;

    protected final transient int indexOfSubtask;

    protected transient StreamOperatorStateHandler stateHandler;

    protected transient InternalTimeServiceManager<?> timeServiceManager;

    protected final int numInputs;

    /** used to stored the cached records. It could be local file system or remote file system. */
    private Path basePath;
    /** file system. */
    protected FileSystem fileSystem;
    /** DataCacheWriter for each input. */
    protected DataCacheWriter[] dataCacheWriters;
    /** segment list for each input. */
    protected List<Segment>[] segmentLists;

    public AbstractBroadcastWrapperOperator(
            StreamOperatorParameters<T> parameters,
            StreamOperatorFactory<T> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation[] inTypes,
            boolean[] isBlocking) {
        this.parameters = Objects.requireNonNull(parameters);
        this.streamConfig = Objects.requireNonNull(parameters.getStreamConfig());
        this.containingTask = Objects.requireNonNull(parameters.getContainingTask());
        this.output = Objects.requireNonNull(parameters.getOutput());
        this.operatorFactory = Objects.requireNonNull(operatorFactory);
        this.metrics = createOperatorMetricGroup(containingTask.getEnvironment(), streamConfig);
        this.wrappedOperator =
                (S)
                        StreamOperatorFactoryUtil.<T, S>createOperator(
                                        operatorFactory,
                                        (StreamTask) parameters.getContainingTask(),
                                        parameters.getStreamConfig(),
                                        parameters.getOutput(),
                                        parameters.getOperatorEventDispatcher())
                                .f0;
        this.mailboxExecutor =
                containingTask.getMailboxExecutorFactory().createExecutor(TaskMailbox.MIN_PRIORITY);
        this.broadcastStreamNames = broadcastStreamNames;
        this.inTypes = inTypes;
        this.isBlocking = isBlocking;
        this.broadcastVariablesReady = false;
        this.indexOfSubtask = containingTask.getIndexInSubtaskGroup();
        this.numInputs = inTypes.length;

        basePath =
                new Path(
                        containingTask
                                .getEnvironment()
                                .getTaskManagerInfo()
                                .getConfiguration()
                                .get(IterationOptions.DATA_CACHE_PATH));
        try {
            fileSystem = basePath.getFileSystem();
            dataCacheWriters = new DataCacheWriter[numInputs];
            for (int i = 0; i < numInputs; i++) {
                dataCacheWriters[i] =
                        new DataCacheWriter(
                                inTypes[i].createSerializer(containingTask.getExecutionConfig()),
                                fileSystem,
                                () ->
                                        new Path(
                                                basePath.toString()
                                                        + "/"
                                                        + "cache-"
                                                        + parameters
                                                                .getStreamConfig()
                                                                .getOperatorID()
                                                                .toHexString()
                                                        + "-"
                                                        + UUID.randomUUID().toString()));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        segmentLists = new List[numInputs];
        for (int i = 0; i < numInputs; i++) {
            segmentLists[i] = new ArrayList<>();
        }
    }

    /**
     * check whether all of broadcast variables are ready.
     *
     * @return
     */
    protected boolean areBroadcastVariablesReady() {
        if (broadcastVariablesReady) {
            return true;
        }
        for (String name : broadcastStreamNames) {
            if (!BroadcastContext.isCacheFinished(Tuple2.of(name, indexOfSubtask))) {
                return false;
            }
        }
        broadcastVariablesReady = true;
        return true;
    }

    private OperatorMetricGroup createOperatorMetricGroup(
            Environment environment, StreamConfig streamConfig) {
        try {
            OperatorMetricGroup operatorMetricGroup =
                    environment
                            .getMetricGroup()
                            .getOrAddOperator(
                                    streamConfig.getOperatorID(), streamConfig.getOperatorName());
            if (streamConfig.isChainEnd()) {
                ((InternalOperatorIOMetricGroup) operatorMetricGroup.getIOMetricGroup())
                        .reuseOutputMetricsForTask();
            }
            return operatorMetricGroup;
        } catch (Exception e) {
            LOG.warn("An error occurred while instantiating task metrics.", e);
            return UnregisteredMetricGroups.createUnregisteredOperatorMetricGroup();
        }
    }

    @Override
    public void open() throws Exception {
        wrappedOperator.open();
    }

    @Override
    public void close() throws Exception {
        wrappedOperator.close();
        for (String name : broadcastStreamNames) {
            BroadcastContext.remove(Tuple2.of(name, indexOfSubtask));
        }
    }

    @Override
    public void finish() throws Exception {
        wrappedOperator.finish();
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        wrappedOperator.prepareSnapshotPreBarrier(checkpointId);
    }

    @Override
    public void initializeState(StreamTaskStateInitializer streamTaskStateManager)
            throws Exception {
        final TypeSerializer<?> keySerializer =
                streamConfig.getStateKeySerializer(containingTask.getUserCodeClassLoader());

        StreamOperatorStateContext streamOperatorStateContext =
                streamTaskStateManager.streamOperatorStateContext(
                        getOperatorID(),
                        getClass().getSimpleName(),
                        parameters.getProcessingTimeService(),
                        this,
                        keySerializer,
                        containingTask.getCancelables(),
                        metrics,
                        streamConfig.getManagedMemoryFractionOperatorUseCaseOfSlot(
                                ManagedMemoryUseCase.STATE_BACKEND,
                                containingTask
                                        .getEnvironment()
                                        .getTaskManagerInfo()
                                        .getConfiguration(),
                                containingTask.getUserCodeClassLoader()),
                        false);
        stateHandler =
                new StreamOperatorStateHandler(
                        streamOperatorStateContext,
                        containingTask.getExecutionConfig(),
                        containingTask.getCancelables());
        stateHandler.initializeOperatorState(this);

        this.timeServiceManager = streamOperatorStateContext.internalTimerServiceManager();

        broadcastVariablesReady = false;

        wrappedOperator.initializeState(
                (operatorID,
                        operatorClassName,
                        processingTimeService,
                        keyContext,
                        keySerializerX,
                        streamTaskCloseableRegistry,
                        metricGroup,
                        managedMemoryFraction,
                        isUsingCustomRawKeyedState) ->
                        new ProxyStreamOperatorStateContext(
                                streamOperatorStateContext, "wrapped-"));
    }

    @Override
    public OperatorSnapshotFutures snapshotState(
            long checkpointId,
            long timestamp,
            CheckpointOptions checkpointOptions,
            CheckpointStreamFactory storageLocation)
            throws Exception {
        return stateHandler.snapshotState(
                this,
                Optional.ofNullable(timeServiceManager),
                streamConfig.getOperatorName(),
                checkpointId,
                timestamp,
                checkpointOptions,
                storageLocation,
                false);
    }

    @Override
    public void initializeState(StateInitializationContext stateInitializationContext)
            throws Exception {
        if (wrappedOperator instanceof StreamOperatorStateHandler.CheckpointedStreamOperator) {
            ((CheckpointedStreamOperator) wrappedOperator)
                    .initializeState(stateInitializationContext);
        }
        List<StatePartitionStreamProvider> inputs =
                IteratorUtils.toList(
                        stateInitializationContext.getRawOperatorStateInputs().iterator());
        Preconditions.checkState(inputs.size() < 2);
        if (inputs.size() == 1) {
            InputStream inputStream = inputs.get(0).getStream();
            DataInputStream dis =
                    new DataInputStream(new NonClosingInputStreamDecorator(inputStream));
            Preconditions.checkState(dis.readInt() == numInputs);
            for (int i = 0; i < numInputs; i++) {
                DataCacheSnapshot dataCacheSnapshot =
                        DataCacheSnapshot.recover(
                                inputStream,
                                fileSystem,
                                () ->
                                        new Path(
                                                basePath.toString()
                                                        + "/"
                                                        + "cache-"
                                                        + parameters
                                                                .getStreamConfig()
                                                                .getOperatorID()
                                                                .toHexString()
                                                        + "-"
                                                        + UUID.randomUUID().toString()));
                segmentLists[i].addAll(dataCacheSnapshot.getSegments());
            }
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext stateSnapshotContext) throws Exception {
        if (wrappedOperator instanceof StreamOperatorStateHandler.CheckpointedStreamOperator) {
            ((CheckpointedStreamOperator) wrappedOperator).snapshotState(stateSnapshotContext);
        }

        OperatorStateCheckpointOutputStream checkpointOutputStream =
                stateSnapshotContext.getRawOperatorStateOutput();
        checkpointOutputStream.startNewPartition();
        try (DataOutputStream dos =
                new DataOutputStream(new NonClosingOutpusStreamDecorator(checkpointOutputStream))) {
            dos.writeInt(numInputs);
        }
        for (int i = 0; i < numInputs; i++) {
            dataCacheWriters[i].finishCurrentSegmentAndStartNewSegment();
            segmentLists[i].addAll(dataCacheWriters[i].getNewlyFinishedSegments());
            DataCacheSnapshot dataCacheSnapshot =
                    new DataCacheSnapshot(fileSystem, null, segmentLists[i]);
            dataCacheSnapshot.writeTo(checkpointOutputStream);
        }
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

    @Override
    public void endInput(int inputId) throws Exception {
        if (wrappedOperator instanceof BoundedOneInput) {
            ((BoundedOneInput) wrappedOperator).endInput();
        }
        if (wrappedOperator instanceof BoundedMultiInput) {
            ((BoundedMultiInput) wrappedOperator).endInput(inputId);
        }
    }
}
