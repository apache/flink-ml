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

package org.apache.flink.ml.iteration.operator.allround;

import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.metrics.groups.OperatorMetricGroup;
import org.apache.flink.ml.iteration.IterationListener;
import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.operator.AbstractWrapperOperator;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
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
import org.apache.flink.util.OutputTag;

import java.io.IOException;

/** The base class for the all-round wrapper operators. */
public abstract class AbstractAllRoundWrapperOperator<T, S extends StreamOperator<T>>
        extends AbstractWrapperOperator<T> {

    protected final S wrappedOperator;

    private final IterationContext iterationContext;

    public AbstractAllRoundWrapperOperator(
            StreamOperatorParameters<IterationRecord<T>> parameters,
            StreamOperatorFactory<T> operatorFactory) {
        super(parameters, operatorFactory);

        this.wrappedOperator =
                (S)
                        StreamOperatorFactoryUtil.<T, S>createOperator(
                                        operatorFactory,
                                        (StreamTask) parameters.getContainingTask(),
                                        parameters.getStreamConfig(),
                                        proxyOutput,
                                        parameters.getOperatorEventDispatcher())
                                .f0;
        this.iterationContext = new IterationContext();
    }

    @Override
    public void onEpochWatermarkIncrement(int epochWatermark) throws IOException {
        if (wrappedOperator instanceof IterationListener) {
            notifyEpochWatermarkIncrement((IterationListener<?>) wrappedOperator, epochWatermark);
        } else if (wrappedOperator instanceof AbstractUdfStreamOperator) {
            Object udf = ((AbstractUdfStreamOperator<?, ?>) wrappedOperator).getUserFunction();
            if (udf instanceof IterationListener) {
                notifyEpochWatermarkIncrement((IterationListener<?>) udf, epochWatermark);
            }
        }

        // Broadcast the events.
        super.onEpochWatermarkIncrement(epochWatermark);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private void notifyEpochWatermarkIncrement(IterationListener<?> listener, int epochWatermark) {
        if (epochWatermark != Integer.MAX_VALUE) {
            listener.onEpochWatermarkIncremented(
                    epochWatermark, iterationContext, (Output) proxyOutput);
        } else {
            listener.onIterationTerminated(iterationContext, (Output) proxyOutput);
        }
    }

    @Override
    public void open() throws Exception {
        wrappedOperator.open();
    }

    @Override
    public void close() throws Exception {
        wrappedOperator.close();
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
    public void endInput() throws Exception {
        if (wrappedOperator instanceof BoundedOneInput) {
            ((BoundedOneInput) wrappedOperator).endInput();
        }
    }

    private class IterationContext implements IterationListener.Context {

        @Override
        public <X> void output(OutputTag<X> outputTag, X value) {
            proxyOutput.collect(outputTag, new StreamRecord<>(value));
        }
    }
}
