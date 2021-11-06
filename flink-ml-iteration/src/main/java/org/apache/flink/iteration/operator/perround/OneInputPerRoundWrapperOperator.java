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

package org.apache.flink.iteration.operator.perround;

import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;
import org.apache.flink.util.FlinkRuntimeException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Per-round wrapper operator for the one-input operator. */
public class OneInputPerRoundWrapperOperator<IN, OUT>
        extends AbstractPerRoundWrapperOperator<OUT, OneInputStreamOperator<IN, OUT>>
        implements OneInputStreamOperator<IterationRecord<IN>, IterationRecord<OUT>> {

    private static final Logger LOG =
            LoggerFactory.getLogger(OneInputPerRoundWrapperOperator.class);

    private final StreamRecord<IN> reusedInput;

    public OneInputPerRoundWrapperOperator(
            StreamOperatorParameters<IterationRecord<OUT>> parameters,
            StreamOperatorFactory<OUT> operatorFactory) {
        super(parameters, operatorFactory);
        this.reusedInput = new StreamRecord<>(null, 0);
    }

    @Override
    protected void endInputAndEmitMaxWatermark(
            OneInputStreamOperator<IN, OUT> operator, int epoch, int epochWatermark)
            throws Exception {
        OperatorUtils.processOperatorOrUdfIfSatisfy(
                operator, BoundedOneInput.class, BoundedOneInput::endInput);
        operator.processWatermark(new Watermark(Long.MAX_VALUE));
    }

    @Override
    public void processElement(StreamRecord<IterationRecord<IN>> element) throws Exception {
        switch (element.getValue().getType()) {
            case RECORD:
                reusedInput.replace(element.getValue().getValue(), element.getTimestamp());
                setIterationContextRound(element.getValue().getEpoch());
                getWrappedOperator(element.getValue().getEpoch()).processElement(reusedInput);
                clearIterationContextRound();
                break;
            case EPOCH_WATERMARK:
                onEpochWatermarkEvent(0, element.getValue());
                break;
            default:
                throw new FlinkRuntimeException("Not supported iteration record type: " + element);
        }
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        processForEachWrappedOperator(
                (round, wrappedOperator) -> wrappedOperator.processWatermark(mark));
    }

    @Override
    public void processWatermarkStatus(WatermarkStatus watermarkStatus) throws Exception {
        processForEachWrappedOperator(
                (round, wrappedOperator) ->
                        wrappedOperator.processWatermarkStatus(watermarkStatus));
    }

    @Override
    public void processLatencyMarker(LatencyMarker latencyMarker) throws Exception {
        reportOrForwardLatencyMarker(latencyMarker);
    }
}
