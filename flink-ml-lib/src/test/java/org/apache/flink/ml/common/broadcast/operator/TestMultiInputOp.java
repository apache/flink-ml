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

import org.apache.flink.streaming.api.operators.AbstractInput;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorV2;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/** Utility class used for testing {@link MultipleInputBroadcastWrapperOperator}. */
public class TestMultiInputOp extends AbstractStreamOperatorV2<Integer>
        implements MultipleInputStreamOperator<Integer>, HasBroadcastVariable {

    private final String[] broadcastNames;

    private final int[] expectedSizes;

    private List<Input> inputList;

    private Map<String, List<?>> broadcastVariables = new HashMap<>();

    public TestMultiInputOp(
            StreamOperatorParameters<Integer> parameters,
            int numberOfInputs,
            String[] broadcastNames,
            int[] expectedSizes) {
        super(parameters, numberOfInputs);
        this.inputList = new ArrayList<>(numberOfInputs);
        for (int i = 0; i < numberOfInputs; i++) {
            inputList.add(new TestMultiInputOp.ProxyInput(this, i + 1));
        }
        this.broadcastNames = broadcastNames;
        this.expectedSizes = expectedSizes;
    }

    @Override
    public void setBroadcastVariable(String name, List<?> broadcastVariable) {
        broadcastVariables.put(name, broadcastVariable);
    }

    @Override
    public List<Input> getInputs() {
        return inputList;
    }

    private class ProxyInput extends AbstractInput<Integer, Integer> {

        public ProxyInput(AbstractStreamOperatorV2<Integer> owner, int inputId) {
            super(owner, inputId);
        }

        @Override
        public void processElement(StreamRecord<Integer> streamRecord) {
            for (int i = 0; i < broadcastNames.length; i++) {
                List<?> source = broadcastVariables.get(broadcastNames[i]);
                assertEquals(expectedSizes[i], source.size());
            }
            output.collect(streamRecord);
        }
    }
}
