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
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;

/** Wrapper for WithBroadcastTwoInputStreamOperator. */
public class TwoInputBroadcastWrapperOperator<IN1, IN2, OUT>
        extends AbstractBroadcastWrapperOperator<OUT, TwoInputStreamOperator<IN1, IN2, OUT>>
        implements TwoInputStreamOperator<IN1, IN2, OUT> {

    public TwoInputBroadcastWrapperOperator(
            StreamOperatorParameters<OUT> parameters,
            StreamOperatorFactory<OUT> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation[] inTypes,
            boolean[] isBlocking) {
        super(parameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
    }

    @Override
    public void processElement1(StreamRecord<IN1> streamRecord) throws Exception {
        if (isBlocking[0]) {
            if (areBroadcastVariablesReady()) {
                dataCacheWriters[0].finishCurrentSegmentAndStartNewSegment();
                segmentLists[0].addAll(dataCacheWriters[0].getNewlyFinishedSegments());
                if (segmentLists[0].size() != 0) {
                    DataCacheReader dataCacheReader =
                            new DataCacheReader<>(
                                    inTypes[0].createSerializer(
                                            containingTask.getExecutionConfig()),
                                    fileSystem,
                                    segmentLists[0]);
                    while (dataCacheReader.hasNext()) {
                        wrappedOperator.processElement1(new StreamRecord(dataCacheReader.next()));
                    }
                }
                segmentLists[0].clear();
                wrappedOperator.processElement1(streamRecord);

            } else {
                dataCacheWriters[0].addRecord(streamRecord.getValue());
            }

        } else {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            wrappedOperator.processElement1(streamRecord);
        }
    }

    @Override
    public void processElement2(StreamRecord<IN2> streamRecord) throws Exception {
        if (isBlocking[1]) {
            if (areBroadcastVariablesReady()) {
                dataCacheWriters[1].finishCurrentSegmentAndStartNewSegment();
                segmentLists[1].addAll(dataCacheWriters[1].getNewlyFinishedSegments());
                if (segmentLists[1].size() != 0) {
                    DataCacheReader dataCacheReader =
                            new DataCacheReader<>(
                                    inTypes[1].createSerializer(
                                            containingTask.getExecutionConfig()),
                                    fileSystem,
                                    segmentLists[1]);
                    while (dataCacheReader.hasNext()) {
                        wrappedOperator.processElement2(new StreamRecord(dataCacheReader.next()));
                    }
                }
                segmentLists[1].clear();
                wrappedOperator.processElement2(streamRecord);

            } else {
                dataCacheWriters[1].addRecord(streamRecord.getValue());
            }

        } else {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            wrappedOperator.processElement2(streamRecord);
        }
    }

    @Override
    public void endInput(int inputId) throws Exception {
        if (inputId == 1) {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            dataCacheWriters[0].finishCurrentSegmentAndStartNewSegment();
            segmentLists[0].addAll(dataCacheWriters[0].getNewlyFinishedSegments());
            if (segmentLists[0].size() != 0) {
                DataCacheReader dataCacheReader =
                        new DataCacheReader(
                                inTypes[0].createSerializer(containingTask.getExecutionConfig()),
                                fileSystem,
                                segmentLists[0]);
                while (dataCacheReader.hasNext()) {
                    wrappedOperator.processElement1(new StreamRecord(dataCacheReader.next()));
                }
                segmentLists[0].clear();
            }
        } else if (inputId == 2) {
            while (!areBroadcastVariablesReady()) {
                mailboxExecutor.yield();
            }
            dataCacheWriters[1].finishCurrentSegmentAndStartNewSegment();
            segmentLists[1].addAll(dataCacheWriters[1].getNewlyFinishedSegments());
            if (segmentLists[1].size() != 0) {
                DataCacheReader dataCacheReader =
                        new DataCacheReader(
                                inTypes[0].createSerializer(containingTask.getExecutionConfig()),
                                fileSystem,
                                segmentLists[1]);
                while (dataCacheReader.hasNext()) {
                    wrappedOperator.processElement2(new StreamRecord(dataCacheReader.next()));
                }
                segmentLists[1].clear();
            }
        }
        super.endInput(inputId);
    }

    @Override
    public void processWatermark1(Watermark watermark) throws Exception {
        wrappedOperator.processWatermark1(watermark);
    }

    @Override
    public void processWatermark2(Watermark watermark) throws Exception {
        wrappedOperator.processWatermark2(watermark);
    }

    @Override
    public void processLatencyMarker1(LatencyMarker latencyMarker) throws Exception {
        wrappedOperator.processLatencyMarker1(latencyMarker);
    }

    @Override
    public void processLatencyMarker2(LatencyMarker latencyMarker) throws Exception {
        wrappedOperator.processLatencyMarker2(latencyMarker);
    }

    @Override
    public void processWatermarkStatus1(WatermarkStatus watermarkStatus) throws Exception {
        wrappedOperator.processWatermarkStatus1(watermarkStatus);
    }

    @Override
    public void processWatermarkStatus2(WatermarkStatus watermarkStatus) throws Exception {
        wrappedOperator.processWatermarkStatus2(watermarkStatus);
    }
}
