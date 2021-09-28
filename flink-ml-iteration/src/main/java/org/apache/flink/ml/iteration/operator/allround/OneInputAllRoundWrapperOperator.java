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

import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;
import org.apache.flink.util.FlinkRuntimeException;

/** All-round wrapper for the one-input operator. */
public class OneInputAllRoundWrapperOperator<IN, OUT>
        extends AbstractAllRoundWrapperOperator<OUT, OneInputStreamOperator<IN, OUT>>
        implements OneInputStreamOperator<IterationRecord<IN>, IterationRecord<OUT>> {

    private final StreamRecord<IN> reusedInput;

    public OneInputAllRoundWrapperOperator(
            StreamOperatorParameters<IterationRecord<OUT>> parameters,
            StreamOperatorFactory<OUT> operatorFactory) {
        super(parameters, operatorFactory);
        this.reusedInput = new StreamRecord<>(null, 0);
    }

    @Override
    public void processElement(StreamRecord<IterationRecord<IN>> element) throws Exception {
        setIterationContextElement(element.getValue());
        switch (element.getValue().getType()) {
            case RECORD:
                setIterationContextElement(element.getValue());
                reusedInput.replace(element.getValue().getValue(), element.getTimestamp());
                wrappedOperator.processElement(reusedInput);
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
        wrappedOperator.processWatermark(mark);
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
    public void setKeyContextElement(StreamRecord<IterationRecord<IN>> record) throws Exception {
        reusedInput.replace(record.getValue().getValue(), record.getTimestamp());
        wrappedOperator.setKeyContextElement(reusedInput);
    }
}
