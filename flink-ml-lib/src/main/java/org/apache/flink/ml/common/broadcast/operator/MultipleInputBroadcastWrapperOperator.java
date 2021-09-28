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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.ml.iteration.datacache.nonkeyed.DataCacheReader;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;

import java.util.ArrayList;
import java.util.List;

/** Wrapper for WithBroadcastMultipleInputStreamOperator. */
public class MultipleInputBroadcastWrapperOperator<OUT>
        extends AbstractBroadcastWrapperOperator<OUT, MultipleInputStreamOperator<OUT>>
        implements MultipleInputStreamOperator<OUT> {

    public MultipleInputBroadcastWrapperOperator(
            StreamOperatorParameters<OUT> parameters,
            StreamOperatorFactory<OUT> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation[] inTypes,
            boolean[] isBlocking) {
        super(parameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
    }

    @Override
    public List<Input> getInputs() {
        List<Input> proxyInputs = new ArrayList<>();
        for (int i = 0; i < wrappedOperator.getInputs().size(); i++) {
            proxyInputs.add(new ProxyInput(i));
        }
        return proxyInputs;
    }

    private <IN> void processElement(StreamRecord streamRecord, Input<IN> input) throws Exception {
        input.processElement(streamRecord);
    }

    private <IN> void processWatermark(Watermark watermark, Input<IN> input) throws Exception {
        input.processWatermark(watermark);
    }

    private <IN> void processLatencyMarker(LatencyMarker latencyMarker, Input<IN> input)
            throws Exception {
        input.processLatencyMarker(latencyMarker);
    }

    private <IN> void setKeyContextElement(StreamRecord streamRecord, Input<IN> input)
            throws Exception {
        input.setKeyContextElement(streamRecord);
    }

    private <IN> void processWatermarkStatus(WatermarkStatus watermarkStatus, Input<IN> input)
            throws Exception {
        input.processWatermarkStatus(watermarkStatus);
    }

    @Override
    public void endInput(int inputId) throws Exception {
        ((ProxyInput) (getInputs().get(inputId - 1))).endInput();
    }

    private class ProxyInput<IN> implements Input<IN> {

        private final int inputIdMinusOne;

        private final Input<IN> input;

        public ProxyInput(int inputIdMinusOne) {
            this.inputIdMinusOne = inputIdMinusOne;
            this.input = wrappedOperator.getInputs().get(inputIdMinusOne);
        }

        @Override
        public void processElement(StreamRecord<IN> streamRecord) throws Exception {
            if (isBlocking[inputIdMinusOne]) {
                if (areBroadcastVariablesReady()) {
                    dataCacheWriters[inputIdMinusOne].finishCurrentSegmentAndStartNewSegment();
                    segmentLists[inputIdMinusOne].addAll(
                            dataCacheWriters[inputIdMinusOne].getNewlyFinishedSegments());
                    if (segmentLists[inputIdMinusOne].size() != 0) {
                        DataCacheReader dataCacheReader =
                                new DataCacheReader<>(
                                        inTypes[inputIdMinusOne].createSerializer(
                                                containingTask.getExecutionConfig()),
                                        fileSystem,
                                        segmentLists[inputIdMinusOne]);
                        while (dataCacheReader.hasNext()) {
                            MultipleInputBroadcastWrapperOperator.this.processElement(
                                    new StreamRecord(dataCacheReader.next()), input);
                        }
                    }
                    segmentLists[inputIdMinusOne].clear();
                    MultipleInputBroadcastWrapperOperator.this.processElement(streamRecord, input);

                } else {
                    dataCacheWriters[inputIdMinusOne].addRecord(streamRecord.getValue());
                }

            } else {
                while (!areBroadcastVariablesReady()) {
                    mailboxExecutor.yield();
                }
                MultipleInputBroadcastWrapperOperator.this.processElement(streamRecord, input);
            }
        }

        @Override
        public void processWatermark(Watermark watermark) throws Exception {
            MultipleInputBroadcastWrapperOperator.this.processWatermark(watermark, input);
        }

        @Override
        public void processWatermarkStatus(WatermarkStatus watermarkStatus) throws Exception {
            MultipleInputBroadcastWrapperOperator.this.processWatermarkStatus(
                    watermarkStatus, input);
        }

        @Override
        public void processLatencyMarker(LatencyMarker latencyMarker) throws Exception {
            MultipleInputBroadcastWrapperOperator.this.processLatencyMarker(latencyMarker, input);
        }

        @Override
        public void setKeyContextElement(StreamRecord<IN> streamRecord) throws Exception {
            MultipleInputBroadcastWrapperOperator.this.setKeyContextElement(streamRecord, input);
        }

        public void endInput() throws Exception {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            dataCacheWriters[inputIdMinusOne].finishCurrentSegmentAndStartNewSegment();
            segmentLists[inputIdMinusOne].addAll(
                    dataCacheWriters[inputIdMinusOne].getNewlyFinishedSegments());
            if (segmentLists[inputIdMinusOne].size() != 0) {
                DataCacheReader dataCacheReader =
                        new DataCacheReader(
                                inTypes[inputIdMinusOne].createSerializer(
                                        containingTask.getExecutionConfig()),
                                fileSystem,
                                segmentLists[inputIdMinusOne]);
                while (dataCacheReader.hasNext()) {
                    MultipleInputBroadcastWrapperOperator.this.processElement(
                            new StreamRecord(dataCacheReader.next()), input);
                }
                segmentLists[inputIdMinusOne].clear();
            }
        }
    }
}
