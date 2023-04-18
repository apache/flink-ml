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
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;

/** Wrapper for {@link OneInputStreamOperator}. */
public class OneInputBroadcastWrapperOperator<IN, OUT>
        extends AbstractBroadcastWrapperOperator<OUT, OneInputStreamOperator<IN, OUT>>
        implements OneInputStreamOperator<IN, OUT>, BoundedOneInput {

    OneInputBroadcastWrapperOperator(
            StreamOperatorParameters<OUT> parameters,
            StreamOperatorFactory<OUT> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation<?>[] inTypes,
            boolean[] isBlocking) {
        super(parameters, operatorFactory, broadcastStreamNames, inTypes, isBlocking);
    }

    @Override
    public void processElement(StreamRecord<IN> streamRecord) throws Exception {
        processElementX(
                streamRecord,
                0,
                wrappedOperator::processElement,
                wrappedOperator::processWatermark,
                wrappedOperator::setKeyContextElement);
    }

    @Override
    public void endInput() throws Exception {
        endInputX(
                0,
                wrappedOperator::processElement,
                wrappedOperator::processWatermark,
                wrappedOperator::setKeyContextElement);
        OperatorUtils.processOperatorOrUdfIfSatisfy(
                wrappedOperator, BoundedOneInput.class, BoundedOneInput::endInput);
    }

    @Override
    public void processWatermark(Watermark watermark) throws Exception {
        processWatermarkX(
                watermark,
                0,
                wrappedOperator::processElement,
                wrappedOperator::processWatermark,
                wrappedOperator::setKeyContextElement);
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
}
