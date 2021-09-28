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

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.OperatorSnapshotFutures;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.StreamTaskStateInitializer;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;

import org.apache.commons.collections.IteratorUtils;

import java.util.ArrayList;
import java.util.List;

/** Wrapper for WithBroadcastOneInputStreamOperator. */
public class OneInputBroadcastWrapperOperator<IN, OUT>
        extends AbstractBroadcastWrapperOperator<OUT, OneInputStreamOperator<IN, OUT>>
        implements OneInputStreamOperator<IN, OUT> {

    private List<IN> cache;

    private ListState<IN> cacheState;

    public OneInputBroadcastWrapperOperator(
            StreamOperatorParameters<OUT> parameters,
            StreamOperatorFactory<OUT> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation[] inTypes,
            boolean[] isBlocking) {
        super(parameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
        this.cache = new ArrayList<>();
    }

    @Override
    public void processElement(StreamRecord<IN> streamRecord) throws Exception {
        if (isBlocking[0]) {
            if (areBroadcastVariablesReady()) {
                for (IN ele : cache) {
                    wrappedOperator.processElement(new StreamRecord<>(ele));
                }
                cache.clear();
                wrappedOperator.processElement(streamRecord);

            } else {
                cache.add(streamRecord.getValue());
            }

        } else {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            wrappedOperator.processElement(streamRecord);
        }
    }

    @Override
    public void endInput(int inputId) throws Exception {
        while (!areBroadcastVariablesReady()) {
            mailboxExecutor.yield();
        }
        for (IN ele : cache) {
            wrappedOperator.processElement(new StreamRecord<>(ele));
        }
        super.endInput(inputId);
    }

    @Override
    public void processWatermark(Watermark watermark) throws Exception {
        wrappedOperator.processWatermark(watermark);
    }

    @Override
    public void processWatermarkStatus(WatermarkStatus watermarkStatus) throws Exception {
        wrappedOperator.processWatermarkStatus(watermarkStatus);
    }

    @Override
    public void processLatencyMarker(LatencyMarker latencyMarker) throws Exception {
        wrappedOperator.processLatencyMarker(latencyMarker);
    }

    @Override
    public void setKeyContextElement(StreamRecord<IN> streamRecord) throws Exception {
        wrappedOperator.setKeyContextElement(streamRecord);
    }

    @Override
    public OperatorSnapshotFutures snapshotState(
            long checkpointId,
            long timestamp,
            CheckpointOptions checkpointOptions,
            CheckpointStreamFactory storageLocation)
            throws Exception {
        cacheState.clear();
        cacheState.addAll(cache);
        return super.snapshotState(checkpointId, timestamp, checkpointOptions, storageLocation);
    }

    @Override
    public void initializeState(StreamTaskStateInitializer streamTaskStateManager)
            throws Exception {
        super.initializeState(streamTaskStateManager);
        cacheState =
                stateBackend.getListState(new ListStateDescriptor<IN>("cache_data", inTypes[0]));
        cache = IteratorUtils.toList(cacheState.get().iterator());
    }
}
