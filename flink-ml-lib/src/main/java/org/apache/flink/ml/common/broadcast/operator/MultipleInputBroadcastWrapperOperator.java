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
import org.apache.flink.ml.iteration.operator.OperatorUtils;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
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

/** Wrapper for {@link MultipleInputStreamOperator} that implements {@link HasBroadcastVariable}. */
public class MultipleInputBroadcastWrapperOperator<OUT>
        extends AbstractBroadcastWrapperOperator<OUT, MultipleInputStreamOperator<OUT>>
        implements MultipleInputStreamOperator<OUT>, BoundedMultiInput {

    private final List<Input> inputList;

    @SuppressWarnings("rawtypes")
    MultipleInputBroadcastWrapperOperator(
            StreamOperatorParameters<OUT> parameters,
            StreamOperatorFactory<OUT> operatorFactory,
            String[] broadcastStreamNames,
            TypeInformation[] inTypes,
            boolean[] isBlocked) {
        super(parameters, operatorFactory, broadcastStreamNames, inTypes, isBlocked);
        inputList = new ArrayList<>();
        for (int i = 0; i < wrappedOperator.getInputs().size(); i++) {
            inputList.add(new ProxyInput(i));
        }
    }

    @Override
    public List<Input> getInputs() {
        return inputList;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void endInput(int inputId) throws Exception {
        endInputX(inputId - 1, wrappedOperator.getInputs().get(inputId - 1)::processElement);
        OperatorUtils.processOperatorOrUdfIfSatisfy(
                wrappedOperator,
                BoundedMultiInput.class,
                boundedMultiInput -> boundedMultiInput.endInput(inputId));
    }

    private class ProxyInput<IN> implements Input<IN> {

        /** input index of this input. */
        private final int inputIndex;

        private final Input input;

        public ProxyInput(int inputIndex) {
            this.inputIndex = inputIndex;
            this.input = wrappedOperator.getInputs().get(inputIndex);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void processElement(StreamRecord<IN> streamRecord) throws Exception {
            MultipleInputBroadcastWrapperOperator.this.processElementX(
                    streamRecord, inputIndex, input::processElement);
        }

        @Override
        public void processWatermark(Watermark watermark) throws Exception {
            input.processWatermark(watermark);
        }

        @Override
        public void processWatermarkStatus(WatermarkStatus watermarkStatus) throws Exception {
            input.processWatermarkStatus(watermarkStatus);
        }

        @Override
        public void processLatencyMarker(LatencyMarker latencyMarker) throws Exception {
            input.processLatencyMarker(latencyMarker);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void setKeyContextElement(StreamRecord<IN> streamRecord) throws Exception {
            input.setKeyContextElement(streamRecord);
        }
    }
}
