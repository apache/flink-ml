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

package org.apache.flink.iteration.operator;

import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.broadcast.BroadcastOutput;
import org.apache.flink.iteration.broadcast.BroadcastOutputFactory;
import org.apache.flink.iteration.progresstrack.OperatorEpochWatermarkTracker;
import org.apache.flink.iteration.progresstrack.OperatorEpochWatermarkTrackerFactory;
import org.apache.flink.iteration.progresstrack.OperatorEpochWatermarkTrackerListener;
import org.apache.flink.iteration.proxy.ProxyOutput;
import org.apache.flink.runtime.execution.Environment;
import org.apache.flink.runtime.metrics.groups.InternalOperatorMetricGroup;
import org.apache.flink.runtime.metrics.groups.UnregisteredMetricGroups;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TimestampedCollector;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.OutputTag;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Objects;
import java.util.function.Supplier;

import static org.apache.flink.util.Preconditions.checkState;

/** The base class of all the wrapper operators. It provides the alignment functionality. */
public abstract class AbstractWrapperOperator<T>
        implements StreamOperator<IterationRecord<T>>,
                OperatorEpochWatermarkTrackerListener,
                BoundedMultiInput {

    private static final Logger LOG = LoggerFactory.getLogger(AbstractWrapperOperator.class);

    protected final StreamOperatorParameters<IterationRecord<T>> parameters;

    protected final StreamConfig streamConfig;

    protected final StreamTask<?, ?> containingTask;

    protected final Output<StreamRecord<IterationRecord<T>>> output;

    protected final StreamOperatorFactory<T> operatorFactory;

    protected final IterationContext iterationContext;

    // --------------- proxy ---------------------------

    protected final ProxyOutput<T> proxyOutput;

    protected final EpochSupplier epochWatermarkSupplier;

    // --------------- Metrics ---------------------------

    /** Metric group for the operator. */
    protected final InternalOperatorMetricGroup metrics;

    // ------------- Iteration Related --------------------

    protected final OperatorEpochWatermarkTracker epochWatermarkTracker;

    protected final String uniqueSenderId;

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
        this.epochWatermarkSupplier = new EpochSupplier();

        this.metrics = createOperatorMetricGroup(containingTask.getEnvironment(), streamConfig);

        this.epochWatermarkTracker =
                OperatorEpochWatermarkTrackerFactory.create(streamConfig, containingTask, this);
        this.uniqueSenderId =
                OperatorUtils.getUniqueSenderId(
                        streamConfig.getOperatorID(), containingTask.getIndexInSubtaskGroup());
        this.eventBroadcastOutput =
                BroadcastOutputFactory.createBroadcastOutput(
                        output, metrics.getIOMetricGroup().getNumRecordsOutCounter());
        this.iterationContext = new IterationContext();
    }

    protected void onEpochWatermarkEvent(int inputIndex, IterationRecord<?> iterationRecord)
            throws IOException {
        checkState(
                iterationRecord.getType() == IterationRecord.Type.EPOCH_WATERMARK,
                "The record " + iterationRecord + " is not epoch watermark.");
        epochWatermarkTracker.onEpochWatermark(
                inputIndex, iterationRecord.getSender(), iterationRecord.getEpoch());
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    protected void notifyEpochWatermarkIncrement(IterationListener<?> listener, int epochWatermark)
            throws Exception {
        if (epochWatermark != Integer.MAX_VALUE) {
            listener.onEpochWatermarkIncremented(
                    epochWatermark,
                    iterationContext,
                    new TimestampedCollector<>((Output) proxyOutput));
        } else {
            listener.onIterationTerminated(
                    iterationContext, new TimestampedCollector<>((Output) proxyOutput));
        }
    }

    @Override
    public void onEpochWatermarkIncrement(int epochWatermark) throws IOException {
        eventBroadcastOutput.broadcastEmit(
                new StreamRecord<>(
                        IterationRecord.newEpochWatermark(epochWatermark, uniqueSenderId)));
    }

    protected void setIterationContextRound(Integer contextRound) {
        proxyOutput.setContextRound(contextRound);
        epochWatermarkSupplier.set(contextRound);
    }

    protected void clearIterationContextRound() {
        proxyOutput.setContextRound(null);
        epochWatermarkSupplier.set(null);
    }

    @Override
    public void endInput(int i) throws Exception {
        epochWatermarkTracker.finish(i - 1);
    }

    private InternalOperatorMetricGroup createOperatorMetricGroup(
            Environment environment, StreamConfig streamConfig) {
        try {
            InternalOperatorMetricGroup operatorMetricGroup =
                    environment
                            .getMetricGroup()
                            .getOrAddOperator(
                                    streamConfig.getOperatorID(), streamConfig.getOperatorName());
            if (streamConfig.isChainEnd()) {
                operatorMetricGroup.getIOMetricGroup().reuseOutputMetricsForTask();
            }
            return operatorMetricGroup;
        } catch (Exception e) {
            LOG.warn("An error occurred while instantiating task metrics.", e);
            return UnregisteredMetricGroups.createUnregisteredOperatorMetricGroup();
        }
    }

    private class IterationContext implements IterationListener.Context {

        @Override
        public <X> void output(OutputTag<X> outputTag, X value) {
            proxyOutput.collect(outputTag, new StreamRecord<>(value));
        }
    }

    private static class EpochSupplier implements Supplier<Integer> {

        private Integer epoch;

        public void set(Integer epoch) {
            this.epoch = epoch;
        }

        @Override
        public Integer get() {
            return epoch;
        }
    }
}
