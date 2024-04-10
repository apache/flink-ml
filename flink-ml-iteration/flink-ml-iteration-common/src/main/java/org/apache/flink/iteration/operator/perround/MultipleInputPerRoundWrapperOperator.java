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
import org.apache.flink.streaming.api.graph.StreamEdge;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;
import org.apache.flink.util.FlinkRuntimeException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Per-round wrapper for the multiple-inputs operator. */
public class MultipleInputPerRoundWrapperOperator<OUT>
        extends AbstractPerRoundWrapperOperator<OUT, MultipleInputStreamOperator<OUT>>
        implements MultipleInputStreamOperator<IterationRecord<OUT>> {

    /** The number of total inputs. */
    private final int numberOfInputs;

    /**
     * Cached inputs for each epoch. This is to avoid repeat calls to the {@link
     * MultipleInputStreamOperator#getInputs()}, which might not returns the same inputs for each
     * call.
     */
    private final Map<Integer, List<Input>> operatorInputsByEpoch = new HashMap<>();

    public MultipleInputPerRoundWrapperOperator(
            StreamOperatorParameters<IterationRecord<OUT>> parameters,
            StreamOperatorFactory<OUT> operatorFactory) {
        super(parameters, operatorFactory);

        // Determine how much inputs we have
        List<StreamEdge> inEdges =
                streamConfig.getInPhysicalEdges(containingTask.getUserCodeClassLoader());
        this.numberOfInputs =
                inEdges.stream().map(StreamEdge::getTypeNumber).collect(Collectors.toSet()).size();
    }

    @Override
    protected MultipleInputStreamOperator<OUT> getWrappedOperator(int epoch) {
        MultipleInputStreamOperator<OUT> operator = super.getWrappedOperator(epoch);
        operatorInputsByEpoch.put(epoch, operator.getInputs());
        return operator;
    }

    @Override
    protected void endInputAndEmitMaxWatermark(
            MultipleInputStreamOperator<OUT> operator, int epoch, int epochWatermark)
            throws Exception {
        OperatorUtils.processOperatorOrUdfIfSatisfy(
                operator,
                BoundedMultiInput.class,
                boundedMultiInput -> {
                    for (int i = 0; i < numberOfInputs; ++i) {
                        boundedMultiInput.endInput(i + 1);
                    }
                });

        for (int i = 0; i < numberOfInputs; ++i) {
            operatorInputsByEpoch.get(epoch).get(i).processWatermark(new Watermark(Long.MAX_VALUE));
        }
    }

    @Override
    protected void closeStreamOperator(
            MultipleInputStreamOperator<OUT> operator, int epoch, int epochWatermark)
            throws Exception {
        super.closeStreamOperator(operator, epoch, epochWatermark);
        operatorInputsByEpoch.remove(epoch);
    }

    @Override
    @SuppressWarnings({"rawtypes"})
    public List<Input> getInputs() {
        List<Input> proxyInputs = new ArrayList<>();

        for (int i = 0; i < numberOfInputs; ++i) {
            // TODO: Note that here we relies on the assumption that the
            // stream graph generator labels the input from 1 to n for
            // the input array, which we map them from 0 to n - 1.
            proxyInputs.add(new ProxyInput(i));
        }
        return proxyInputs;
    }

    private class ProxyInput<IN> implements Input<IterationRecord<IN>> {

        private final int inputIndex;

        private final StreamRecord<IN> reusedInput;

        public ProxyInput(int inputIndex) {
            this.inputIndex = inputIndex;
            this.reusedInput = new StreamRecord<>(null, 0);
        }

        @Override
        public void processElement(StreamRecord<IterationRecord<IN>> element) throws Exception {
            switch (element.getValue().getType()) {
                case RECORD:
                    // Ensures the operators are created.
                    getWrappedOperator(element.getValue().getEpoch());
                    reusedInput.replace(element.getValue().getValue(), element.getTimestamp());
                    setIterationContextRound(element.getValue().getEpoch());
                    operatorInputsByEpoch
                            .get(element.getValue().getEpoch())
                            .get(inputIndex)
                            .processElement(reusedInput);
                    clearIterationContextRound();
                    break;
                case EPOCH_WATERMARK:
                    onEpochWatermarkEvent(inputIndex, element.getValue());
                    break;
                default:
                    throw new FlinkRuntimeException(
                            "Not supported iteration record type: " + element);
            }
        }

        @Override
        public void processWatermark(Watermark mark) throws Exception {
            processForEachWrappedOperator(
                    (round, wrappedOperator) -> {
                        operatorInputsByEpoch.get(round).get(inputIndex).processWatermark(mark);
                    });
        }

        @Override
        public void processWatermarkStatus(WatermarkStatus watermarkStatus) throws Exception {
            processForEachWrappedOperator(
                    (round, wrappedOperator) -> {
                        operatorInputsByEpoch
                                .get(round)
                                .get(inputIndex)
                                .processWatermarkStatus(watermarkStatus);
                    });
        }

        @Override
        public void processLatencyMarker(LatencyMarker latencyMarker) throws Exception {
            reportOrForwardLatencyMarker(latencyMarker);
        }

        @Override
        public void setKeyContextElement(StreamRecord<IterationRecord<IN>> element)
                throws Exception {

            if (element.getValue().getType() == IterationRecord.Type.RECORD) {
                // Ensures the operators are created.
                getWrappedOperator(element.getValue().getEpoch());
                reusedInput.replace(element.getValue().getValue(), element.getTimestamp());
                operatorInputsByEpoch
                        .get(element.getValue().getEpoch())
                        .get(inputIndex)
                        .setKeyContextElement(reusedInput);
            }
        }
    }
}
