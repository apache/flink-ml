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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.proxy.state.ProxyStreamOperatorStateContext;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.ml.common.broadcast.typeinfo.CacheElement;
import org.apache.flink.ml.common.broadcast.typeinfo.CacheElementSerializer;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.execution.Environment;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.metrics.groups.InternalOperatorIOMetricGroup;
import org.apache.flink.runtime.metrics.groups.UnregisteredMetricGroups;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.graph.StreamConfig;
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
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.CloseableIterator;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.ThrowingConsumer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Queue;

/** Base class for the shared objects wrapper operators. */
abstract class AbstractSharedObjectsWrapperOperator<
                T, S extends AbstractSharedObjectsStreamOperator<T>>
        implements StreamOperator<T>, IterationListener<T>, CheckpointedStreamOperator {

    private static final Logger LOG =
            LoggerFactory.getLogger(AbstractSharedObjectsWrapperOperator.class);

    protected final StreamOperatorParameters<T> parameters;

    protected final StreamConfig streamConfig;

    protected final StreamTask<?, ?> containingTask;

    protected final Output<StreamRecord<T>> output;

    protected final StreamOperatorFactory<T> operatorFactory;

    protected final OperatorMetricGroup metrics;

    protected final S wrappedOperator;

    private final SharedObjectsContextImpl context;
    private final int numInputs;
    private final TypeSerializer<?>[] inTypeSerializers;
    private final ListStateWithCache<CacheElement<?>>[] cachedElements;
    private final Queue<ReadRequest<?>>[] readRequests;
    private final boolean[] hasCachedElements;

    protected transient StreamOperatorStateHandler stateHandler;

    protected transient InternalTimeServiceManager<?> timeServiceManager;

    @SuppressWarnings({"unchecked", "rawtypes"})
    AbstractSharedObjectsWrapperOperator(
            StreamOperatorParameters<T> parameters,
            StreamOperatorFactory<T> operatorFactory,
            SharedObjectsContextImpl context) {
        this.parameters = Objects.requireNonNull(parameters);
        this.streamConfig = Objects.requireNonNull(parameters.getStreamConfig());
        this.containingTask = Objects.requireNonNull(parameters.getContainingTask());
        this.output = Objects.requireNonNull(parameters.getOutput());
        this.operatorFactory = Objects.requireNonNull(operatorFactory);
        this.context = context;
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
        wrappedOperator.onSharedObjectsContextSet(context);

        StreamConfig.InputConfig[] inputConfigs =
                streamConfig.getInputs(containingTask.getUserCodeClassLoader());
        int numNetworkInputs = 0;
        while (numNetworkInputs < inputConfigs.length
                && inputConfigs[numNetworkInputs] instanceof StreamConfig.NetworkInputConfig) {
            numNetworkInputs++;
        }
        numInputs = numNetworkInputs;

        inTypeSerializers = new TypeSerializer[numInputs];
        readRequests = new Queue[numInputs];
        for (int i = 0; i < numInputs; i++) {
            inTypeSerializers[i] =
                    streamConfig.getTypeSerializerIn(i, containingTask.getUserCodeClassLoader());
            readRequests[i] = new ArrayDeque<>(getInputReadRequests(i));
        }
        cachedElements = new ListStateWithCache[numInputs];
        hasCachedElements = new boolean[numInputs];
        Arrays.fill(hasCachedElements, false);
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
     * Checks if the read requests are satisfied for the input.
     *
     * @param inputId The input id, starting from 0.
     * @param wait Whether to wait until all requests satisfied, or not.
     * @return If all requests of this input are satisfied.
     */
    private boolean checkReadRequestsReady(int inputId, boolean wait) {
        Queue<ReadRequest<?>> requests = readRequests[inputId];
        while (!requests.isEmpty()) {
            ReadRequest<?> request = requests.poll();
            try {
                if (null == context.read(request, wait)) {
                    requests.add(request);
                    return false;
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        return true;
    }

    /**
     * Gets {@link ReadRequest}s required for processing elements in the input.
     *
     * @param inputId The input id, starting from 0.
     * @return The {@link ReadRequest}s required for processing elements.
     */
    protected abstract List<ReadRequest<?>> getInputReadRequests(int inputId);

    /**
     * Extracts common processing logic in subclasses' processing elements.
     *
     * @param streamRecord The input record.
     * @param inputId The input id, starting from 0.
     * @param elementConsumer The consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer The consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @param keyContextSetter The consumer function of setting key context, i.e.,
     *     operator.setKeyContext(...).
     * @throws Exception Possible exception.
     */
    @SuppressWarnings({"rawtypes"})
    protected void processElementX(
            StreamRecord streamRecord,
            int inputId,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer,
            ThrowingConsumer<StreamRecord, Exception> keyContextSetter)
            throws Exception {
        if (checkReadRequestsReady(inputId, false)) {
            if (hasCachedElements[inputId]) {
                processCachedElements(
                        inputId, elementConsumer, watermarkConsumer, keyContextSetter);
                hasCachedElements[inputId] = false;
            }
            keyContextSetter.accept(streamRecord);
            elementConsumer.accept(streamRecord);
        } else {
            cachedElements[inputId].add(CacheElement.newRecord(streamRecord.getValue()));
            hasCachedElements[inputId] = true;
        }
    }

    /**
     * Extracts common processing logic in subclasses' processing watermarks.
     *
     * @param watermark The input watermark.
     * @param inputId The input id, starting from 0.
     * @param elementConsumer The consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer The consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @param keyContextSetter The consumer function of setting key context, i.e.,
     *     operator.setKeyContext(...).
     * @throws Exception Possible exception.
     */
    @SuppressWarnings({"rawtypes"})
    protected void processWatermarkX(
            Watermark watermark,
            int inputId,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer,
            ThrowingConsumer<StreamRecord, Exception> keyContextSetter)
            throws Exception {
        if (checkReadRequestsReady(inputId, false)) {
            if (hasCachedElements[inputId]) {
                processCachedElements(
                        inputId, elementConsumer, watermarkConsumer, keyContextSetter);
                hasCachedElements[inputId] = false;
            }
            watermarkConsumer.accept(watermark);
        } else {
            cachedElements[inputId].add(CacheElement.newWatermark(watermark.getTimestamp()));
            hasCachedElements[inputId] = true;
        }
    }

    /**
     * Extracts common processing logic in subclasses' endInput(...).
     *
     * @param inputId The input id, starting from 0.
     * @param elementConsumer The consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer The consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @param keyContextSetter The consumer function of setting key context, i.e.,
     *     operator.setKeyContext(...).
     * @throws Exception Possible exception.
     */
    @SuppressWarnings("rawtypes")
    protected void endInputX(
            int inputId,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer,
            ThrowingConsumer<StreamRecord, Exception> keyContextSetter)
            throws Exception {
        if (hasCachedElements[inputId]) {
            checkReadRequestsReady(inputId, true);
            processCachedElements(inputId, elementConsumer, watermarkConsumer, keyContextSetter);
            hasCachedElements[inputId] = false;
        }
    }

    /**
     * Processes elements that are cached by {@link ListStateWithCache}.
     *
     * @param inputId The input id, starting from 0.
     * @param elementConsumer The consumer function of StreamRecord, i.e.,
     *     operator.processElement(...).
     * @param watermarkConsumer The consumer function of WaterMark, i.e.,
     *     operator.processWatermark(...).
     * @param keyContextSetter The consumer function of setting key context, i.e.,
     *     operator.setKeyContext(...).
     * @throws Exception Possible exception.
     */
    @SuppressWarnings({"rawtypes", "unchecked"})
    private void processCachedElements(
            int inputId,
            ThrowingConsumer<StreamRecord, Exception> elementConsumer,
            ThrowingConsumer<Watermark, Exception> watermarkConsumer,
            ThrowingConsumer<StreamRecord, Exception> keyContextSetter)
            throws Exception {
        for (CacheElement<?> cacheElement : cachedElements[inputId].get()) {
            switch (cacheElement.getType()) {
                case RECORD:
                    StreamRecord record = new StreamRecord(cacheElement.getRecord());
                    keyContextSetter.accept(record);
                    elementConsumer.accept(record);
                    break;
                case WATERMARK:
                    watermarkConsumer.accept(new Watermark(cacheElement.getWatermark()));
                    break;
                default:
                    throw new RuntimeException(
                            "Unsupported CacheElement type: " + cacheElement.getType());
            }
        }
        cachedElements[inputId].clear();
        Preconditions.checkState(readRequests[inputId].isEmpty());
        readRequests[inputId].addAll(getInputReadRequests(inputId));
    }

    @Override
    public void open() throws Exception {
        wrappedOperator.open();
    }

    @Override
    public void close() throws Exception {
        wrappedOperator.close();
        context.clear();
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
    @SuppressWarnings("unchecked, rawtypes")
    public void initializeState(StateInitializationContext stateInitializationContext)
            throws Exception {
        StreamingRuntimeContext runtimeContext = wrappedOperator.getRuntimeContext();
        context.initializeState(wrappedOperator, runtimeContext, stateInitializationContext);
        for (int i = 0; i < numInputs; i++) {
            cachedElements[i] =
                    new ListStateWithCache<>(
                            new CacheElementSerializer(inTypeSerializers[i]),
                            containingTask,
                            runtimeContext,
                            stateInitializationContext,
                            streamConfig.getOperatorID());
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext stateSnapshotContext) throws Exception {
        context.snapshotState(stateSnapshotContext);
        wrappedOperator.snapshotState(stateSnapshotContext);
        for (int i = 0; i < numInputs; i++) {
            cachedElements[i].snapshotState(stateSnapshotContext);
        }
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
    public void setKeyContextElement1(StreamRecord record) throws Exception {
        wrappedOperator.setKeyContextElement1(record);
    }

    @Override
    public void setKeyContextElement2(StreamRecord record) throws Exception {
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
    public Object getCurrentKey() {
        return wrappedOperator.getCurrentKey();
    }

    @Override
    public void setCurrentKey(Object key) {
        wrappedOperator.setCurrentKey(key);
    }

    protected abstract void processCachedElementsBeforeEpochIncremented(int inputId)
            throws Exception;

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<T> collector) throws Exception {
        for (int i = 0; i < numInputs; i += 1) {
            processCachedElementsBeforeEpochIncremented(i);
        }
        this.context.incStep(epochWatermark);
        if (wrappedOperator instanceof IterationListener) {
            //noinspection unchecked
            ((IterationListener<T>) wrappedOperator)
                    .onEpochWatermarkIncremented(epochWatermark, context, collector);
        }
    }

    @Override
    public void onIterationTerminated(Context context, Collector<T> collector) throws Exception {
        this.context.incStep();
        if (wrappedOperator instanceof IterationListener) {
            //noinspection unchecked
            ((IterationListener<T>) wrappedOperator).onIterationTerminated(context, collector);
        }
    }
}
