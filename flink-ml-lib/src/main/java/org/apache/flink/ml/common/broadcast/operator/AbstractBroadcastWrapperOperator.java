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
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.ml.common.broadcast.BroadcastContext;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.execution.Environment;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.metrics.groups.InternalOperatorIOMetricGroup;
import org.apache.flink.runtime.metrics.groups.UnregisteredMetricGroups;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.runtime.state.OperatorStateBackend;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorV2;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OperatorSnapshotFutures;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorFactoryUtil;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.StreamTaskStateInitializer;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.streaming.runtime.tasks.mailbox.TaskMailbox;
import org.apache.flink.util.FlinkRuntimeException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/** Base class for the broadcast wrapper operators. */
public abstract class AbstractBroadcastWrapperOperator<T, S extends StreamOperator<T>>
        implements StreamOperator<T>, BoundedMultiInput {

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

    protected OperatorStateBackend stateBackend;

    protected final transient int indexOfSubtask;

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
    }

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
    public OperatorSnapshotFutures snapshotState(
            long checkpointId,
            long timestamp,
            CheckpointOptions checkpointOptions,
            CheckpointStreamFactory storageLocation)
            throws Exception {
        return wrappedOperator.snapshotState(
                checkpointId, timestamp, checkpointOptions, storageLocation);
    }

    @Override
    public void initializeState(StreamTaskStateInitializer streamTaskStateManager)
            throws Exception {
        wrappedOperator.initializeState(streamTaskStateManager);
        if (wrappedOperator instanceof AbstractStreamOperator) {
            stateBackend = ((AbstractStreamOperator) wrappedOperator).getOperatorStateBackend();
        } else if (wrappedOperator instanceof AbstractStreamOperatorV2) {
            stateBackend = ((AbstractStreamOperatorV2) wrappedOperator).getOperatorStateBackend();
        } else {
            throw new FlinkRuntimeException(
                    "The operator must be either "
                            + AbstractStreamOperator.class.getSimpleName()
                            + " or "
                            + AbstractStreamOperatorV2.class.getSimpleName());
        }
        broadcastVariablesReady = false;
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
