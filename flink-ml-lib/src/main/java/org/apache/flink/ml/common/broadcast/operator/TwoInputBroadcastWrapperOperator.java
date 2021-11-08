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
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;

/** Wrapper for {@link TwoInputStreamOperator}. */
public class TwoInputBroadcastWrapperOperator<IN1, IN2, OUT>
        extends AbstractBroadcastWrapperOperator<OUT, TwoInputStreamOperator<IN1, IN2, OUT>>
        implements TwoInputStreamOperator<IN1, IN2, OUT>, BoundedMultiInput {

    TwoInputBroadcastWrapperOperator(
            StreamOperatorParameters<OUT> parameters,
            StreamOperatorFactory<OUT> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation<?>[] inTypes,
            boolean[] isBlocking) {
        super(parameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
    }

    @Override
    public void processElement1(StreamRecord<IN1> streamRecord) throws Exception {
        processElementX(
                streamRecord,
                0,
                wrappedOperator::processElement1,
                wrappedOperator::processWatermark1);
    }

    @Override
    public void processElement2(StreamRecord<IN2> streamRecord) throws Exception {
        processElementX(
                streamRecord,
                1,
                wrappedOperator::processElement2,
                wrappedOperator::processWatermark2);
    }

    @Override
    public void endInput(int inputId) throws Exception {
        if (inputId == 1) {
            endInputX(
                    inputId - 1,
                    wrappedOperator::processElement1,
                    wrappedOperator::processWatermark1);
        } else {
            endInputX(
                    inputId - 1,
                    wrappedOperator::processElement2,
                    wrappedOperator::processWatermark2);
        }
        OperatorUtils.processOperatorOrUdfIfSatisfy(
                wrappedOperator,
                BoundedMultiInput.class,
                boundedMultipleInput -> boundedMultipleInput.endInput(inputId));
    }

    @Override
    public void processWatermark1(Watermark watermark) throws Exception {
        processWatermarkX(
                watermark, 0, wrappedOperator::processElement1, wrappedOperator::processWatermark1);
    }

    @Override
    public void processWatermark2(Watermark watermark) throws Exception {
        processWatermarkX(
                watermark, 1, wrappedOperator::processElement2, wrappedOperator::processWatermark2);
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
