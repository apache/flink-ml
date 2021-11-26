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

import org.apache.flink.api.common.functions.RichFunction;
import org.apache.flink.api.common.operators.MailboxExecutor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.iteration.datacache.nonkeyed.DataCacheReader;
import org.apache.flink.iteration.datacache.nonkeyed.DataCacheSnapshot;
import org.apache.flink.iteration.datacache.nonkeyed.DataCacheWriter;
import org.apache.flink.iteration.datacache.nonkeyed.Segment;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.iteration.proxy.state.ProxyStreamOperatorStateContext;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.ml.common.broadcast.BroadcastContext;
import org.apache.flink.ml.common.broadcast.BroadcastStreamingRuntimeContext;
import org.apache.flink.ml.common.broadcast.typeinfo.CacheElement;
import org.apache.flink.ml.common.broadcast.typeinfo.CacheElementTypeInfo;
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
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
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
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.streaming.runtime.tasks.mailbox.TaskMailbox;
import org.apache.flink.util.CloseableIterator;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.ThrowingConsumer;

import org.apache.commons.collections.IteratorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/** Base class for the broadcast wrapper operators. */
public abstract class AbstractBroadcastWrapperOperator<T, S extends StreamOperator<T>>
        implements StreamOperator<T>, StreamOperatorStateHandler.CheckpointedStreamOperator {

    private static final Logger LOG =
            LoggerFactory.getLogger(AbstractBroadcastWrapperOperator.class);

    protected final StreamOperatorParameters<T> parameters;

    protected final StreamConfig streamConfig;

    protected final StreamTask<?, ?> containingTask;

    protected final Output<StreamRecord<T>> output;

    protected final StreamOperatorFactory<T> operatorFactory;

    protected final OperatorMetricGroup metrics;

    protected final S wrappedOperator;

    protected transient StreamOperatorStateHandler stateHandler;

    protected transient InternalTimeServiceManager<?> timeServiceManager;

    protected final MailboxExecutor mailboxExecutor;

    /** variables specific for withBroadcast functionality. */
    protected final String[] broadcastStreamNames;

    /**
     * whether each input is blocked. Inputs with broadcast variables can only process their input
     * records after broadcast variables are ready. One input is non-blocked if it can consume its
     * inputs (by caching) when broadcast variables are not ready. Otherwise it has to block the
     * processing and wait until the broadcast variables are ready to be accessed.
     */
    protected final boolean[] isBlocked;

    /** type information of each input. */
    protected final TypeInformation<?>[] inTypes;

    /** whether all broadcast variables of this operator are ready. */
    protected boolean broadcastVariablesReady;

    /** index of this subtask. */
    protected final transient int indexOfSubtask;

    /** number of the inputs of this operator. */
    protected final int numInputs;

    /** runtimeContext of the rich function in wrapped operator. */
    BroadcastStreamingRuntimeContext wrappedOperatorRuntimeContext;

    /**
     * path of the file used to stored the cached records. It could be local file system or remote
     * file system.
     */
    private Path basePath;

    /** DataCacheWriter for each input. */
    @SuppressWarnings("rawtypes")
    protected DataCacheWriter[] dataCacheWriters;

    /** whether each input has pending elements. */
    protected boolean[] hasPendingElements;

    @SuppressWarnings({"unchecked", "rawtypes"})
    AbstractBroadcastWrapperOperator(
            StreamOperatorParameters<T> parameters,
            StreamOperatorFactory<T> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation<?>[] inTypes,
            boolean[] isBlocked) {
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
                                        (StreamTask) containingTask,
                                        streamConfig,
                                        output,
                                        parameters.getOperatorEventDispatcher())
                                .f0;

        boolean hasRichFunction =
                wrappedOperator instanceof AbstractUdfStreamOperator
                        && ((AbstractUdfStreamOperator) wrappedOperator).getUserFunction()
                                instanceof RichFunction;

        if (hasRichFunction) {
            wrappedOperatorRuntimeContext =
                    new BroadcastStreamingRuntimeContext(
                            containingTask.getEnvironment(),
                            containingTask.getEnvironment().getAccumulatorRegistry().getUserMap(),
                            wrappedOperator.getMetricGroup(),
                            wrappedOperator.getOperatorID(),
                            ((AbstractUdfStreamOperator) wrappedOperator)
                                    .getProcessingTimeService(),
                            null,
                            containingTask.getEnvironment().getExternalResourceInfoProvider());

            ((RichFunction) ((AbstractUdfStreamOperator) wrappedOperator).getUserFunction())
                    .setRuntimeContext(wrappedOperatorRuntimeContext);
        } else {
            throw new RuntimeException(
                    "The operator is not a instance of "
                            + AbstractUdfStreamOperator.class.getSimpleName()
                            + " that contains a "
                            + RichFunction.class.getSimpleName());
        }

        this.mailboxExecutor =
                containingTask.getMailboxExecutorFactory().createExecutor(TaskMailbox.MIN_PRIORITY);
        // variables specific for withBroadcast functionality.
        this.broadcastStreamNames = broadcastStreamNames;
        this.isBlocked = isBlocked;
        this.inTypes = inTypes;
        this.broadcastVariablesReady = false;
        this.indexOfSubtask = containingTask.getIndexInSubtaskGroup();
        this.numInputs = inTypes.length;

        // puts in mailboxExecutor
        for (String name : broadcastStreamNames) {
            BroadcastContext.putMailBoxExecutor(name + "-" + indexOfSubtask, mailboxExecutor);
        }

        basePath =
                OperatorUtils.getDataCachePath(
                        containingTask.getEnvironment().getTaskManagerInfo().getConfiguration(),
                        containingTask
                                .getEnvironment()
                                .getIOManager()
                                .getSpillingDirectoriesPaths());
        dataCacheWriters = new DataCacheWriter[numInputs];
        hasPendingElements = new boolean[numInputs];
        Arrays.fill(hasPendingElements, true);
    }

    /**
     * checks whether all of broadcast variables are ready. Besides it maintains a state
     * {broadcastVariablesReady} to avoiding invoking {@code BroadcastContext.isCacheFinished(...)}
     * repeatedly. Finally, it sets broadcast variables for {wrappedOperatorRuntimeContext} if the
     * broadcast variables are ready.
     *
     * @return true if all broadcast variables are ready, false otherwise.
     */
    protected boolean areBroadcastVariablesReady() {
        if (broadcastVariablesReady) {
            return true;
        }
        for (String name : broadcastStreamNames) {
            if (!BroadcastContext.isCacheFinished(name + "-" + indexOfSubtask)) {
                return false;
            } else {
                String key = name + "-" + indexOfSubtask;
                String userKey = name.substring(name.indexOf('-') + 1);
                wrappedOperatorRuntimeContext.setBroadcastVariable(
                        userKey, BroadcastContext.getBroadcastVariable(key));
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

    /**
     * extracts common processing logic in subclasses' processing elements.
     *
     * @param streamRecord the input record.
     * @param inputIndex input id, starts from zero.
     * @param elementConsumer the consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer the consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @throws Exception possible exception.
     */
    @SuppressWarnings({"rawtypes", "unchecked"})
    protected void processElementX(
            StreamRecord streamRecord,
            int inputIndex,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer)
            throws Exception {
        if (!isBlocked[inputIndex]) {
            if (areBroadcastVariablesReady()) {
                if (hasPendingElements[inputIndex]) {
                    processPendingElementsAndWatermarks(
                            inputIndex, elementConsumer, watermarkConsumer);
                    hasPendingElements[inputIndex] = false;
                }
                elementConsumer.accept(streamRecord);

            } else {
                dataCacheWriters[inputIndex].addRecord(
                        CacheElement.newRecord(streamRecord.getValue()));
            }

        } else {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            elementConsumer.accept(streamRecord);
        }
    }

    /**
     * extracts common processing logic in subclasses' processing watermarks.
     *
     * @param watermark the input watermark.
     * @param inputIndex input id, starts from zero.
     * @param elementConsumer the consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer the consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @throws Exception possible exception.
     */
    @SuppressWarnings({"rawtypes", "unchecked"})
    protected void processWatermarkX(
            Watermark watermark,
            int inputIndex,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer)
            throws Exception {
        if (!isBlocked[inputIndex]) {
            if (areBroadcastVariablesReady()) {
                if (hasPendingElements[inputIndex]) {
                    processPendingElementsAndWatermarks(
                            inputIndex, elementConsumer, watermarkConsumer);
                    hasPendingElements[inputIndex] = false;
                }
                watermarkConsumer.accept(watermark);

            } else {
                dataCacheWriters[inputIndex].addRecord(
                        CacheElement.newWatermark(watermark.getTimestamp()));
            }

        } else {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            watermarkConsumer.accept(watermark);
        }
    }

    /**
     * extracts common processing logic in subclasses' endInput(...).
     *
     * @param inputIndex input id, starts from zero.
     * @param elementConsumer the consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer the consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @throws Exception possible exception.
     */
    @SuppressWarnings("rawtypes")
    protected void endInputX(
            int inputIndex,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer)
            throws Exception {
        while (!areBroadcastVariablesReady()) {
            mailboxExecutor.yield();
        }
        if (hasPendingElements[inputIndex]) {
            processPendingElementsAndWatermarks(inputIndex, elementConsumer, watermarkConsumer);
            hasPendingElements[inputIndex] = false;
        }
    }

    /**
     * processes the pending elements that are cached by {@link DataCacheWriter}.
     *
     * @param inputIndex input id, starts from zero.
     * @param elementConsumer the consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer the consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @throws Exception possible exception.
     */
    @SuppressWarnings({"rawtypes", "unchecked"})
    private void processPendingElementsAndWatermarks(
            int inputIndex,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer)
            throws Exception {
        dataCacheWriters[inputIndex].finishCurrentSegment();
        List<Segment> pendingSegments = dataCacheWriters[inputIndex].getFinishSegments();
        if (pendingSegments.size() != 0) {
            DataCacheReader dataCacheReader =
                    new DataCacheReader<>(
                            new CacheElementTypeInfo<>(inTypes[inputIndex])
                                    .createSerializer(containingTask.getExecutionConfig()),
                            basePath.getFileSystem(),
                            pendingSegments);
            while (dataCacheReader.hasNext()) {
                CacheElement cacheElement = (CacheElement) dataCacheReader.next();
                switch (cacheElement.getType()) {
                    case RECORD:
                        elementConsumer.accept(new StreamRecord(cacheElement.getRecord()));
                        break;
                    case WATERMARK:
                        watermarkConsumer.accept(new Watermark(cacheElement.getWatermark()));
                        break;
                    default:
                        throw new RuntimeException(
                                "Unsupported CacheElement type: " + cacheElement.getType());
                }
            }
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
            BroadcastContext.remove(name + "-" + indexOfSubtask);
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

        timeServiceManager = streamOperatorStateContext.internalTimerServiceManager();

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
                                streamOperatorStateContext,
                                "wrapped-",
                                CloseableIterator.empty(),
                                0));
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
    @SuppressWarnings("unchecked, rawtypes")
    public void initializeState(StateInitializationContext stateInitializationContext)
            throws Exception {
        List<StatePartitionStreamProvider> inputs =
                IteratorUtils.toList(
                        stateInitializationContext.getRawOperatorStateInputs().iterator());
        Preconditions.checkState(
                inputs.size() < 2, "The input from raw operator state should be one or zero.");
        if (inputs.size() == 0) {
            for (int i = 0; i < numInputs; i++) {
                dataCacheWriters[i] =
                        new DataCacheWriter(
                                new CacheElementTypeInfo<>(inTypes[i])
                                        .createSerializer(containingTask.getExecutionConfig()),
                                basePath.getFileSystem(),
                                OperatorUtils.createDataCacheFileGenerator(
                                        basePath, "cache", streamConfig.getOperatorID()));
            }
        } else {
            InputStream inputStream = inputs.get(0).getStream();
            DataInputStream dis =
                    new DataInputStream(new NonClosingInputStreamDecorator(inputStream));
            Preconditions.checkState(dis.readInt() == numInputs, "Number of input is wrong.");
            for (int i = 0; i < numInputs; i++) {
                DataCacheSnapshot dataCacheSnapshot =
                        DataCacheSnapshot.recover(
                                inputStream,
                                basePath.getFileSystem(),
                                OperatorUtils.createDataCacheFileGenerator(
                                        basePath, "cache", streamConfig.getOperatorID()));
                dataCacheWriters[i] =
                        new DataCacheWriter(
                                new CacheElementTypeInfo<>(inTypes[i])
                                        .createSerializer(containingTask.getExecutionConfig()),
                                basePath.getFileSystem(),
                                OperatorUtils.createDataCacheFileGenerator(
                                        basePath, "cache", streamConfig.getOperatorID()),
                                dataCacheSnapshot.getSegments());
            }
        }
    }

    @SuppressWarnings("unchecked")
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
            dataCacheWriters[i].finishCurrentSegment();
            DataCacheSnapshot dataCacheSnapshot =
                    new DataCacheSnapshot(
                            basePath.getFileSystem(),
                            null,
                            dataCacheWriters[i].getFinishSegments());
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
}
