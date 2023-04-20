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

package org.apache.flink.iteration.operator.allround;

import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;
import org.apache.flink.util.FlinkRuntimeException;
import org.apache.flink.util.function.ThrowingConsumer;

/** All-round wrapper for the two-inputs operator. */
public class TwoInputAllRoundWrapperOperator<IN1, IN2, OUT>
        extends AbstractAllRoundWrapperOperator<OUT, TwoInputStreamOperator<IN1, IN2, OUT>>
        implements TwoInputStreamOperator<
                        IterationRecord<IN1>, IterationRecord<IN2>, IterationRecord<OUT>>,
                BoundedMultiInput {

    private final StreamRecord<IN1> reusedInput1;

    private final StreamRecord<IN2> reusedInput2;

    public TwoInputAllRoundWrapperOperator(
            StreamOperatorParameters<IterationRecord<OUT>> parameters,
            StreamOperatorFactory<OUT> operatorFactory) {
        super(parameters, operatorFactory);
        this.reusedInput1 = new StreamRecord<>(null, 0);
        this.reusedInput2 = new StreamRecord<>(null, 0);
    }

    @Override
    public void processElement1(StreamRecord<IterationRecord<IN1>> element) throws Exception {
        processElement(element, 0, reusedInput1, wrappedOperator::processElement1);
    }

    @Override
    public void processElement2(StreamRecord<IterationRecord<IN2>> element) throws Exception {
        processElement(element, 1, reusedInput2, wrappedOperator::processElement2);
    }

    private <IN> void processElement(
            StreamRecord<IterationRecord<IN>> element,
            int inputIndex,
            StreamRecord<IN> reusedInput,
            ThrowingConsumer<StreamRecord<IN>, Exception> processor)
            throws Exception {

        switch (element.getValue().getType()) {
            case RECORD:
                reusedInput.replace(element.getValue().getValue(), element.getTimestamp());
                setIterationContextRound(element.getValue().getEpoch());
                processor.accept(reusedInput);
                clearIterationContextRound();
                break;
            case EPOCH_WATERMARK:
                onEpochWatermarkEvent(inputIndex, element.getValue());
                break;
            default:
                throw new FlinkRuntimeException("Not supported iteration record type: " + element);
        }
    }

    @Override
    public void processWatermark1(Watermark mark) throws Exception {
        wrappedOperator.processWatermark1(mark);
    }

    @Override
    public void processWatermark2(Watermark mark) throws Exception {
        wrappedOperator.processWatermark2(mark);
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

    @Override
    public void endInput(int i) throws Exception {
        super.endInput(i);

        if (wrappedOperator instanceof BoundedMultiInput) {
            setIterationContextRound(Integer.MAX_VALUE);
            ((BoundedMultiInput) wrappedOperator).endInput(i);
            clearIterationContextRound();
        }
    }
}
