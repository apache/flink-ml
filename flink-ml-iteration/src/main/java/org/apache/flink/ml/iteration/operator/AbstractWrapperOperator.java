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

package org.apache.flink.ml.iteration.operator;

import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.broadcast.BroadcastOutput;
import org.apache.flink.ml.iteration.broadcast.BroadcastOutputFactory;
import org.apache.flink.ml.iteration.progresstrack.ProgressTracker;
import org.apache.flink.ml.iteration.progresstrack.ProgressTrackerFactory;
import org.apache.flink.ml.iteration.progresstrack.ProgressTrackerListener;
import org.apache.flink.ml.iteration.proxy.ProxyOutput;
import org.apache.flink.runtime.execution.Environment;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.runtime.metrics.groups.InternalOperatorIOMetricGroup;
import org.apache.flink.runtime.metrics.groups.UnregisteredMetricGroups;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Objects;

import static org.apache.flink.util.Preconditions.checkState;

/** The base class of all the wrapper operators. It provides the alignment functionality. */
public abstract class AbstractWrapperOperator<T>
        implements StreamOperator<IterationRecord<T>>, ProgressTrackerListener, BoundedOneInput {

    private static final Logger LOG = LoggerFactory.getLogger(AbstractWrapperOperator.class);

    protected final StreamOperatorParameters<IterationRecord<T>> parameters;

    protected final StreamConfig streamConfig;

    protected final StreamTask<?, ?> containingTask;

    protected final Output<StreamRecord<IterationRecord<T>>> output;

    protected final StreamOperatorFactory<T> operatorFactory;

    // --------------- proxy ---------------------------

    protected final ProxyOutput<T> proxyOutput;

    // --------------- Metrics ---------------------------

    /** Metric group for the operator. */
    protected final OperatorMetricGroup metrics;

    // ------------- Iteration Related --------------------

    protected final ProgressTracker progressTracker;

    protected final BroadcastOutput<IterationRecord<T>> eventBroadcastOutput;

    public AbstractWrapperOperator(
            StreamOperatorParameters<IterationRecord<T>> parameters,
            StreamOperatorFactory<T> operatorFactory) {
        this.parameters = Objects.requireNonNull(parameters);
        this.streamConfig = Objects.requireNonNull(parameters.getStreamConfig());
        this.containingTask = Objects.requireNonNull(parameters.getContainingTask());
        this.output = Objects.requireNonNull(parameters.getOutput());
        this.operatorFactory = Objects.requireNonNull(operatorFactory);

        this.proxyOutput = new ProxyOutput<>(output);

        this.metrics = createOperatorMetricGroup(containingTask.getEnvironment(), streamConfig);

        this.progressTracker = ProgressTrackerFactory.create(streamConfig, containingTask, this);
        this.eventBroadcastOutput =
                BroadcastOutputFactory.createBroadcastOutput(
                        output, metrics.getIOMetricGroup().getNumRecordsOutCounter());
    }

    protected void onEpochWatermarkEvent(int inputIndex, IterationRecord<?> iterationRecord)
            throws IOException {
        checkState(
                iterationRecord.getType() == IterationRecord.Type.EPOCH_WATERMARK,
                "The record " + iterationRecord + " is not epoch watermark.");
        progressTracker.onEpochWatermark(
                inputIndex, iterationRecord.getSender(), iterationRecord.getRound());
    }

    @Override
    public void onEpochWatermarkIncrement(int epochWatermark) throws IOException {
        eventBroadcastOutput.broadcastEmit(
                new StreamRecord<>(
                        IterationRecord.newEpochWatermark(
                                epochWatermark,
                                OperatorUtils.getUniqueSenderId(
                                        streamConfig.getOperatorID(),
                                        containingTask
                                                .getEnvironment()
                                                .getTaskInfo()
                                                .getIndexOfThisSubtask()))));
    }

    protected void setIterationContextElement(IterationRecord<?> iterationRecord) {
        proxyOutput.setContextElement(iterationRecord);
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
                ((InternalOperatorIOMetricGroup) operatorMetricGroup.getIOMetricGroup()).reuseOutputMetricsForTask();
            }
            return operatorMetricGroup;
        } catch (Exception e) {
            LOG.warn("An error occurred while instantiating task metrics.", e);
            return UnregisteredMetricGroups.createUnregisteredOperatorMetricGroup();
        }
    }
}
