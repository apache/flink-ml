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

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.broadcast.BroadcastContext;
import org.apache.flink.ml.iteration.config.IterationOptions;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.streaming.api.operators.AbstractInput;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorV2;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.MultipleInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;
import org.apache.flink.streaming.util.TestHarnessUtil;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import static org.junit.Assert.*;

public class MultipleInputBroadcastWrapperOperatorTest {
    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();
    private static final String[] broadcastNames = new String[] {"source1", "source2"};
    private static final TypeInformation[] typeInformations =
            new TypeInformation[] {BasicTypeInfo.INT_TYPE_INFO, BasicTypeInfo.INT_TYPE_INFO};
    private static List<Integer> source1 = new ArrayList<>();
    private static List<Integer> source2 = new ArrayList<>();

    @Before
    public void setup() {
        source1.add(1);
        source2.add(1);
        source2.add(2);
    }

    @Test
    public void testProcessElementsAndEpochWatermarks() throws Exception {
        TestMultiInputOpFactory multiOpFactory = new TestMultiInputOpFactory(2);
        BroadcastWrapper broadcastWrapper = new BroadcastWrapper(broadcastNames, typeInformations);
        BroadcastWrapperOperatorFactory wrapperFactory =
                new BroadcastWrapperOperatorFactory(multiOpFactory, broadcastWrapper);
        OperatorID operatorId = new OperatorID();

        try (StreamTaskMailboxTestHarness<Integer> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                MultipleInputStreamTask::new, BasicTypeInfo.INT_TYPE_INFO)
                        .addInput(typeInformations[0])
                        .addInput(typeInformations[1])
                        .setupOutputForSingletonOperatorChain(wrapperFactory, operatorId)
                        .buildUnrestored()) {
            harness.getStreamTask()
                    .getEnvironment()
                    .getTaskManagerInfo()
                    .getConfiguration()
                    .set(
                            IterationOptions.DATA_CACHE_PATH,
                            "file://" + tempFolder.newFolder().getAbsolutePath());
            harness.getStreamTask().restore();
            BroadcastContext.putBroadcastVariable(
                    Tuple2.of(broadcastNames[0], 0), Tuple2.of(true, source1));
            BroadcastContext.putBroadcastVariable(
                    Tuple2.of(broadcastNames[1], 0), Tuple2.of(true, source2));

            Queue<Object> expectedOutput = new ConcurrentLinkedQueue<>();
            for (int i = 0; i < 5; ++i) {
                harness.processElement(new StreamRecord<>(i, 1000), 0);
                harness.processElement(new StreamRecord<>(i, 1000), 1);
                expectedOutput.add(new StreamRecord<>(i, 1000));
                expectedOutput.add(new StreamRecord<>(i, 1000));
            }
            TestHarnessUtil.assertOutputEquals(
                    "Output was not correct", expectedOutput, harness.getOutput());
        }
    }

    /** Factory class for {@link TestMultiInputOp}. */
    private static class TestMultiInputOpFactory extends AbstractStreamOperatorFactory<Long> {
        private int numInputs;

        public TestMultiInputOpFactory(int numInputs) {
            this.numInputs = numInputs;
        }

        @Override
        public <T extends StreamOperator<Long>> T createStreamOperator(
                StreamOperatorParameters<Long> streamOperatorParameters) {
            return (T) new TestMultiInputOp(streamOperatorParameters, numInputs);
        }

        @Override
        public Class<? extends StreamOperator> getStreamOperatorClass(ClassLoader classLoader) {
            return TestMultiInputOp.class;
        }
    }

    /**
     * The MultiInputStreamOperator that checks the size of the broadcast inputs and pass the input
     * to downstream operator directly.
     */
    private static class TestMultiInputOp extends AbstractStreamOperatorV2<Long>
            implements MultipleInputStreamOperator<Long> {
        private List<Input> inputList;

        public TestMultiInputOp(StreamOperatorParameters<Long> parameters, int numberOfInputs) {
            super(parameters, numberOfInputs);
            this.inputList = new ArrayList<>(numberOfInputs);
            for (int i = 0; i < numberOfInputs; i++) {
                inputList.add(new ProxyInput(this, i + 1));
            }
        }

        @Override
        public List<Input> getInputs() {
            return inputList;
        }

        private class ProxyInput extends AbstractInput<Long, Long> {

            public ProxyInput(AbstractStreamOperatorV2<Long> owner, int inputId) {
                super(owner, inputId);
            }

            @Override
            public void processElement(StreamRecord<Long> streamRecord) throws Exception {
                List<Long> bcSource1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
                List<Long> bcSource2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
                assertEquals(source1, bcSource1);
                assertEquals(source2, bcSource2);
                output.collect(streamRecord);
            }
        }
    }
}
