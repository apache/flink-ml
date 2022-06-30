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

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.MetricOptions;
import org.apache.flink.contrib.streaming.state.RocksDBKeyedStateBackend;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.AbstractWrapperOperator;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.iteration.proxy.state.ProxyStateSnapshotContext;
import org.apache.flink.iteration.proxy.state.ProxyStreamOperatorStateContext;
import org.apache.flink.iteration.utils.ReflectionUtils;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.metrics.groups.UnregisteredMetricGroups;
import org.apache.flink.runtime.state.AbstractKeyedStateBackend;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.runtime.state.DefaultOperatorStateBackend;
import org.apache.flink.runtime.state.KeyedStateBackend;
import org.apache.flink.runtime.state.OperatorStateBackend;
import org.apache.flink.runtime.state.OperatorStateCheckpointOutputStream;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StatePartitionStreamProvider;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.state.heap.HeapKeyedStateBackend;
import org.apache.flink.streaming.api.operators.InternalTimeServiceManager;
import org.apache.flink.streaming.api.operators.OperatorSnapshotFutures;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactoryUtil;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.StreamOperatorStateContext;
import org.apache.flink.streaming.api.operators.StreamOperatorStateHandler;
import org.apache.flink.streaming.api.operators.StreamTaskStateInitializer;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.streaming.util.LatencyStats;
import org.apache.flink.util.CloseableIterable;
import org.apache.flink.util.ExceptionUtils;
import org.apache.flink.util.InstantiationUtil;
import org.apache.flink.util.function.BiConsumerWithException;

import org.apache.commons.collections.IteratorUtils;
import org.rocksdb.RocksDB;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import static org.apache.flink.util.Preconditions.checkState;

/** The base class for all the per-round wrapper operators. */
public abstract class AbstractPerRoundWrapperOperator<T, S extends StreamOperator<T>>
        extends AbstractWrapperOperator<T>
        implements StreamOperatorStateHandler.CheckpointedStreamOperator {

    private static final Logger LOG =
            LoggerFactory.getLogger(AbstractPerRoundWrapperOperator.class);

    private static final String HEAP_KEYED_STATE_NAME =
            "org.apache.flink.runtime.state.heap.HeapKeyedStateBackend";

    private static final String ROCKSDB_KEYED_STATE_NAME =
            "org.apache.flink.contrib.streaming.state.RocksDBKeyedStateBackend";

    /** The wrapped operators for each round. */
    private final Map<Integer, S> wrappedOperators;

    protected final LatencyStats latencyStats;

    private transient StreamOperatorStateContext streamOperatorStateContext;

    private transient StreamOperatorStateHandler stateHandler;

    private transient InternalTimeServiceManager<?> timeServiceManager;

    private transient KeySelector<?, ?> stateKeySelector1;

    private transient KeySelector<?, ?> stateKeySelector2;

    private int latestEpochWatermark = -1;

    // --------------- state ---------------------------

    /** Stores the parallelism of the last run so that we could check if parallelism is changed. */
    private ListState<Integer> parallelismState;

    /** Stores the latest epoch watermark received. */
    private ListState<Integer> latestEpochWatermarkState;

    /**
     * Stores the list of epochs that are not aligned yet. After failover the wrapped operators
     * corresponding to these epochs would be re-crated.
     */
    private ListState<Integer> pendingEpochState;

    private ListState<Integer> rawStateEpochState;

    public AbstractPerRoundWrapperOperator(
            StreamOperatorParameters<IterationRecord<T>> parameters,
            StreamOperatorFactory<T> operatorFactory) {
        super(parameters, operatorFactory);

        this.wrappedOperators = new HashMap<>();
        this.latencyStats = initializeLatencyStats();
    }

    protected S getWrappedOperator(int round) {
        return getWrappedOperator(
                round, CloseableIterable.<StatePartitionStreamProvider>empty().iterator(), 0);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private S getWrappedOperator(
            int round, Iterator<StatePartitionStreamProvider> rawOperatorStates, int count) {
        S wrappedOperator = wrappedOperators.get(round);
        if (wrappedOperator != null) {
            return wrappedOperator;
        }

        // We need to clone the operator factory to also support SimpleOperatorFactory.
        try {
            StreamOperatorFactory<T> clonedOperatorFactory =
                    InstantiationUtil.clone(
                            operatorFactory, containingTask.getUserCodeClassLoader());
            wrappedOperator =
                    (S)
                            StreamOperatorFactoryUtil.<T, S>createOperator(
                                            clonedOperatorFactory,
                                            (StreamTask) parameters.getContainingTask(),
                                            OperatorUtils.createWrappedOperatorConfig(
                                                    parameters.getStreamConfig()),
                                            proxyOutput,
                                            parameters.getOperatorEventDispatcher())
                                    .f0;
            initializeStreamOperator(wrappedOperator, round, rawOperatorStates, count);
            wrappedOperators.put(round, wrappedOperator);
            return wrappedOperator;
        } catch (Exception e) {
            ExceptionUtils.rethrow(e);
        }

        return wrappedOperator;
    }

    protected abstract void endInputAndEmitMaxWatermark(S operator, int epoch, int epochWatermark)
            throws Exception;

    protected void closeStreamOperator(S operator, int epoch, int epochWatermark) throws Exception {
        setIterationContextRound(epoch);
        OperatorUtils.processOperatorOrUdfIfSatisfy(
                operator,
                IterationListener.class,
                listener -> notifyEpochWatermarkIncrement(listener, epochWatermark));
        endInputAndEmitMaxWatermark(operator, epoch, epochWatermark);
        operator.finish();
        operator.close();
        setIterationContextRound(null);

        // Cleanup the states used by this operator.
        cleanupOperatorStates(epoch);

        if (stateHandler.getKeyedStateBackend() != null) {
            cleanupKeyedStates(epoch);
        }
    }

    @Override
    public void onEpochWatermarkIncrement(int epochWatermark) throws IOException {
        checkState(epochWatermark >= 0, "The epoch watermark should be non-negative.");
        if (epochWatermark > latestEpochWatermark) {
            latestEpochWatermark = epochWatermark;

            // Destroys all the operators with round < epoch watermark. Notes that
            // the onEpochWatermarkIncrement must be from 0 and increment by 1 each time, except
            // for the last round.
            try {
                if (epochWatermark < Integer.MAX_VALUE) {
                    S wrappedOperator = wrappedOperators.remove(epochWatermark);
                    if (wrappedOperator != null) {
                        closeStreamOperator(wrappedOperator, epochWatermark, epochWatermark);
                    }
                } else {
                    List<Integer> sortedEpochs = new ArrayList<>(wrappedOperators.keySet());
                    Collections.sort(sortedEpochs);
                    for (Integer epoch : sortedEpochs) {
                        closeStreamOperator(wrappedOperators.remove(epoch), epoch, epochWatermark);
                    }
                }
            } catch (Exception exception) {
                ExceptionUtils.rethrow(exception);
            }
        }

        super.onEpochWatermarkIncrement(epochWatermark);
    }

    protected void processForEachWrappedOperator(
            BiConsumerWithException<Integer, S, Exception> consumer) throws Exception {
        for (Map.Entry<Integer, S> entry : wrappedOperators.entrySet()) {
            consumer.accept(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public void open() throws Exception {}

    @Override
    public void initializeState(StreamTaskStateInitializer streamTaskStateManager)
            throws Exception {
        final TypeSerializer<?> keySerializer =
                streamConfig.getStateKeySerializer(containingTask.getUserCodeClassLoader());

        streamOperatorStateContext =
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
                        isUsingCustomRawKeyedState());

        stateHandler =
                new StreamOperatorStateHandler(
                        streamOperatorStateContext,
                        containingTask.getExecutionConfig(),
                        containingTask.getCancelables());
        stateHandler.initializeOperatorState(this);
        this.timeServiceManager = streamOperatorStateContext.internalTimerServiceManager();

        stateKeySelector1 =
                streamConfig.getStatePartitioner(0, containingTask.getUserCodeClassLoader());
        stateKeySelector2 =
                streamConfig.getStatePartitioner(1, containingTask.getUserCodeClassLoader());
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        parallelismState =
                context.getOperatorStateStore()
                        .getUnionListState(
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
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("latestEpoch", IntSerializer.INSTANCE));
        OperatorStateUtils.getUniqueElement(latestEpochWatermarkState, "latestEpoch")
                .ifPresent(
                        oldLatestEpochWatermark -> latestEpochWatermark = oldLatestEpochWatermark);

        // Notes that the list must be sorted.
        rawStateEpochState =
                context.getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("rawStateEpoch", Integer.class));
        List<Integer> rawStateEpochs = IteratorUtils.toList(rawStateEpochState.get().iterator());

        // Notes that the list must be sorted.
        pendingEpochState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("pendingEpochs", IntSerializer.INSTANCE));
        List<Integer> pendingEpochs = IteratorUtils.toList(pendingEpochState.get().iterator());

        // Unfortunately, for the raw state we could not call get input stream unless the previous
        // records are consumed. We would have to do a "merge" of the two lists.
        Iterator<StatePartitionStreamProvider> rawStates =
                context.getRawOperatorStateInputs().iterator();

        int nextRawStateEntryIndex = 0;
        for (int epoch : pendingEpochs) {
            checkState(
                    nextRawStateEntryIndex == rawStateEpochs.size()
                            || rawStateEpochs.get(nextRawStateEntryIndex) >= epoch,
                    String.format(
                            "Unexpected raw state indices %s and epochs %s",
                            rawStateEpochs.toString(), pendingEpochs.toString()));
            // Let's find how much entries this epoch has.
            int numberOfStateEntries = 0;
            while (nextRawStateEntryIndex < rawStateEpochs.size()
                    && rawStateEpochs.get(nextRawStateEntryIndex) == epoch) {
                numberOfStateEntries++;
                nextRawStateEntryIndex++;
            }

            // We first open these operators
            getWrappedOperator(epoch, rawStates, numberOfStateEntries);
        }
    }

    @Internal
    protected boolean isUsingCustomRawKeyedState() {
        return false;
    }

    @Override
    public void finish() throws Exception {
        checkState(
                wrappedOperators.size() == 0,
                "Some wrapped operators are still not closed yet: " + wrappedOperators.keySet());
    }

    @Override
    public void close() throws Exception {
        if (stateHandler != null) {
            stateHandler.dispose();
        }
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        for (Map.Entry<Integer, S> entry : wrappedOperators.entrySet()) {
            entry.getValue().prepareSnapshotPreBarrier(checkpointId);
        }
    }

    @Override
    public OperatorSnapshotFutures snapshotState(
            long checkpointId,
            long timestamp,
            CheckpointOptions checkpointOptions,
            CheckpointStreamFactory factory)
            throws Exception {
        return stateHandler.snapshotState(
                this,
                Optional.ofNullable(timeServiceManager),
                streamConfig.getOperatorName(),
                checkpointId,
                timestamp,
                checkpointOptions,
                factory,
                isUsingCustomRawKeyedState());
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        OperatorStateCheckpointOutputStream rawOperatorStateOutputStream =
                context.getRawOperatorStateOutput();
        List<Integer> operatorStateEpoch = new ArrayList<>();

        List<Integer> sortedEpochs = new ArrayList<>(wrappedOperators.keySet());
        Collections.sort(sortedEpochs);

        for (int epoch : sortedEpochs) {
            S wrappedOperator = wrappedOperators.get(epoch);
            if (StreamOperatorStateHandler.CheckpointedStreamOperator.class.isAssignableFrom(
                    wrappedOperator.getClass())) {
                ((StreamOperatorStateHandler.CheckpointedStreamOperator) wrappedOperator)
                        .snapshotState(new ProxyStateSnapshotContext(context));

                // Gets the count of the raw operator state.
                int numberOfPartitions = rawOperatorStateOutputStream.getNumberOfPartitions();
                while (operatorStateEpoch.size() < numberOfPartitions) {
                    operatorStateEpoch.add(epoch);
                }
            }
        }

        // Then snapshot our own states
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

        // The list must be sorted
        rawStateEpochState.update(operatorStateEpoch);

        // The list must be sorted
        pendingEpochState.update(sortedEpochs);
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public void setKeyContextElement1(StreamRecord record) throws Exception {
        setKeyContextElement(record, stateKeySelector1);
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public void setKeyContextElement2(StreamRecord record) throws Exception {
        setKeyContextElement(record, stateKeySelector2);
    }

    private <T> void setKeyContextElement(StreamRecord<T> record, KeySelector<T, ?> selector)
            throws Exception {
        if (selector != null
                && ((IterationRecord<?>) record.getValue()).getType()
                        == IterationRecord.Type.RECORD) {
            Object key = selector.getKey(record.getValue());
            setCurrentKey(key);
        }
    }

    @Override
    public OperatorMetricGroup getMetricGroup() {
        return metrics;
    }

    @Override
    public OperatorID getOperatorID() {
        return streamConfig.getOperatorID();
    }

    @Override
    public void notifyCheckpointComplete(long l) throws Exception {
        for (Map.Entry<Integer, S> entry : wrappedOperators.entrySet()) {
            entry.getValue().notifyCheckpointComplete(l);
        }
    }

    @Override
    public void notifyCheckpointAborted(long checkpointId) throws Exception {
        for (Map.Entry<Integer, S> entry : wrappedOperators.entrySet()) {
            entry.getValue().notifyCheckpointAborted(checkpointId);
        }
    }

    @Override
    public void setCurrentKey(Object key) {
        stateHandler.setCurrentKey(key);
    }

    @Override
    public Object getCurrentKey() {
        if (stateHandler == null) {
            return null;
        }

        return stateHandler.getCurrentKey();
    }

    protected void reportOrForwardLatencyMarker(LatencyMarker marker) {
        // all operators are tracking latencies
        this.latencyStats.reportLatency(marker);

        // everything except sinks forwards latency markers
        this.output.emitLatencyMarker(marker);
    }

    private LatencyStats initializeLatencyStats() {
        try {
            Configuration taskManagerConfig =
                    containingTask.getEnvironment().getTaskManagerInfo().getConfiguration();
            int historySize = taskManagerConfig.getInteger(MetricOptions.LATENCY_HISTORY_SIZE);
            if (historySize <= 0) {
                LOG.warn(
                        "{} has been set to a value equal or below 0: {}. Using default.",
                        MetricOptions.LATENCY_HISTORY_SIZE,
                        historySize);
                historySize = MetricOptions.LATENCY_HISTORY_SIZE.defaultValue();
            }

            final String configuredGranularity =
                    taskManagerConfig.getString(MetricOptions.LATENCY_SOURCE_GRANULARITY);
            LatencyStats.Granularity granularity;
            try {
                granularity =
                        LatencyStats.Granularity.valueOf(
                                configuredGranularity.toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException iae) {
                granularity = LatencyStats.Granularity.OPERATOR;
                LOG.warn(
                        "Configured value {} option for {} is invalid. Defaulting to {}.",
                        configuredGranularity,
                        MetricOptions.LATENCY_SOURCE_GRANULARITY.key(),
                        granularity);
            }
            MetricGroup jobMetricGroup = this.metrics.getJobMetricGroup();
            return new LatencyStats(
                    jobMetricGroup.addGroup("latency"),
                    historySize,
                    containingTask.getIndexInSubtaskGroup(),
                    getOperatorID(),
                    granularity);
        } catch (Exception e) {
            LOG.warn("An error occurred while instantiating latency metrics.", e);
            return new LatencyStats(
                    UnregisteredMetricGroups.createUnregisteredTaskManagerJobMetricGroup()
                            .addGroup("latency"),
                    1,
                    0,
                    new OperatorID(),
                    LatencyStats.Granularity.SINGLE);
        }
    }

    private void initializeStreamOperator(
            S operator,
            int round,
            Iterator<StatePartitionStreamProvider> rawOperatorStates,
            int count)
            throws Exception {
        operator.initializeState(
                (operatorID,
                        operatorClassName,
                        processingTimeService,
                        keyContext,
                        keySerializer,
                        streamTaskCloseableRegistry,
                        metricGroup,
                        managedMemoryFraction,
                        isUsingCustomRawKeyedState) ->
                        new ProxyStreamOperatorStateContext(
                                streamOperatorStateContext,
                                getRoundStatePrefix(round),
                                rawOperatorStates,
                                count));
        operator.open();
    }

    private void cleanupOperatorStates(int round) {
        String roundPrefix = getRoundStatePrefix(round);
        OperatorStateBackend operatorStateBackend = stateHandler.getOperatorStateBackend();

        if (operatorStateBackend instanceof DefaultOperatorStateBackend) {
            for (String fieldNames :
                    new String[] {
                        "registeredOperatorStates",
                        "registeredBroadcastStates",
                        "accessedStatesByName",
                        "accessedBroadcastStatesByName"
                    }) {
                Map<String, ?> field =
                        ReflectionUtils.getFieldValue(
                                operatorStateBackend,
                                DefaultOperatorStateBackend.class,
                                fieldNames);
                field.entrySet().removeIf(entry -> entry.getKey().startsWith(roundPrefix));
            }
        } else {
            LOG.warn("Unable to cleanup the operator state {}", operatorStateBackend);
        }
    }

    private void cleanupKeyedStates(int round) {
        String roundPrefix = getRoundStatePrefix(round);
        KeyedStateBackend<?> keyedStateBackend = stateHandler.getKeyedStateBackend();
        if (keyedStateBackend.getClass().getName().equals(HEAP_KEYED_STATE_NAME)) {
            ReflectionUtils.<Map<String, ?>>getFieldValue(
                            keyedStateBackend, HeapKeyedStateBackend.class, "registeredKVStates")
                    .entrySet()
                    .removeIf(entry -> entry.getKey().startsWith(roundPrefix));
            ReflectionUtils.<Map<String, ?>>getFieldValue(
                            keyedStateBackend,
                            AbstractKeyedStateBackend.class,
                            "keyValueStatesByName")
                    .entrySet()
                    .removeIf(entry -> entry.getKey().startsWith(roundPrefix));
        } else if (keyedStateBackend.getClass().getName().equals(ROCKSDB_KEYED_STATE_NAME)) {
            RocksDB db =
                    ReflectionUtils.getFieldValue(
                            keyedStateBackend, RocksDBKeyedStateBackend.class, "db");
            HashMap<String, RocksDBKeyedStateBackend.RocksDbKvStateInfo> kvStateInformation =
                    ReflectionUtils.getFieldValue(
                            keyedStateBackend,
                            RocksDBKeyedStateBackend.class,
                            "kvStateInformation");
            kvStateInformation.entrySet().stream()
                    .filter(entry -> entry.getKey().startsWith(roundPrefix))
                    .forEach(
                            entry -> {
                                try {
                                    db.dropColumnFamily(entry.getValue().columnFamilyHandle);
                                } catch (Exception e) {
                                    LOG.error(
                                            "Failed to drop state {} for round {}",
                                            entry.getKey(),
                                            round);
                                }
                            });
            kvStateInformation.entrySet().removeIf(entry -> entry.getKey().startsWith(roundPrefix));

            Map<String, ?> field =
                    ReflectionUtils.getFieldValue(
                            keyedStateBackend,
                            AbstractKeyedStateBackend.class,
                            "keyValueStatesByName");
            field.entrySet().removeIf(entry -> entry.getKey().startsWith(roundPrefix));
        } else {
            LOG.warn("Unable to cleanup the keyed state {}", keyedStateBackend);
        }
    }

    private String getRoundStatePrefix(int round) {
        return "r" + round + "-";
    }

    int getLatestEpochWatermark() {
        return latestEpochWatermark;
    }

    public Map<Integer, S> getWrappedOperators() {
        return wrappedOperators;
    }
}
