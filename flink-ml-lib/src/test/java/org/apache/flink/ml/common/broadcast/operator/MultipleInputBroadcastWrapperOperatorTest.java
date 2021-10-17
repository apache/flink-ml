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
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.MultipleInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;
import org.apache.flink.streaming.util.TestHarnessUtil;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/** Tests the {@link MultipleInputBroadcastWrapperOperator}. */
public class MultipleInputBroadcastWrapperOperatorTest {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private static final String[] BROADCAST_NAMES = new String[] {"source1", "source2"};

    private static final TypeInformation<?>[] TYPE_INFORMATIONS =
            new TypeInformation[] {BasicTypeInfo.INT_TYPE_INFO, BasicTypeInfo.INT_TYPE_INFO};

    private static final List<Integer> SOURCE_1 = Collections.singletonList(1);

    private static final List<Integer> SOURCE_2 = Arrays.asList(1, 2, 3);

    @Test
    public void testProcessElementsAndEpochWatermarks() throws Exception {
        TestMultiInputOpFactory multiOpFactory =
                new TestMultiInputOpFactory(2, BROADCAST_NAMES, new int[] {1, 3});
        BroadcastWrapper<Integer> broadcastWrapper =
                new BroadcastWrapper<>(BROADCAST_NAMES, TYPE_INFORMATIONS);
        BroadcastWrapperOperatorFactory<Integer> wrapperFactory =
                new BroadcastWrapperOperatorFactory<>(multiOpFactory, broadcastWrapper);
        OperatorID operatorId = new OperatorID();

        try (StreamTaskMailboxTestHarness<Integer> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                MultipleInputStreamTask::new, BasicTypeInfo.INT_TYPE_INFO)
                        .addInput(TYPE_INFORMATIONS[0])
                        .addInput(TYPE_INFORMATIONS[1])
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
                    BROADCAST_NAMES[0] + "-" + 0, Tuple2.of(true, SOURCE_1));
            BroadcastContext.putBroadcastVariable(
                    BROADCAST_NAMES[1] + "-" + 0, Tuple2.of(true, SOURCE_2));

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
}
