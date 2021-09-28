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
import org.apache.flink.ml.common.broadcast.BroadcastContext;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.MultipleInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class CacheStreamOperatorTest {
    private static final String[] broadcastNames = new String[] {"source1", "source2"};
    private static final TypeInformation[] typeInformations =
            new TypeInformation[] {BasicTypeInfo.INT_TYPE_INFO, BasicTypeInfo.INT_TYPE_INFO};

    @Test
    public void testCacheStreamOperator() throws Exception {
        OperatorID operatorId = new OperatorID();

        try (StreamTaskMailboxTestHarness<Integer> harness =
                new StreamTaskMailboxTestHarnessBuilder<>(
                                MultipleInputStreamTask::new, BasicTypeInfo.INT_TYPE_INFO)
                        .addInput(BasicTypeInfo.INT_TYPE_INFO)
                        .addInput(BasicTypeInfo.INT_TYPE_INFO)
                        .setupOutputForSingletonOperatorChain(
                                new CacheStreamOperatorFactory<>(broadcastNames, typeInformations),
                                operatorId)
                        .build()) {
            harness.processElement(new StreamRecord<>(1, 2), 0);
            harness.processElement(new StreamRecord<>(2, 3), 0);
            harness.processElement(new StreamRecord<>(3, 2), 1);
            harness.processElement(new StreamRecord<>(4, 2), 1);
            harness.processElement(new StreamRecord<>(5, 3), 1);
            List<Integer> cache1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
            List<Integer> cache2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
            // check broadcast inputs before task finishes.
            assertEquals(null, cache1);
            assertEquals(null, cache2);

            harness.waitForTaskCompletion();
            cache1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
            cache2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
            // check broadcast inputs after task finishes.
            assertEquals(Arrays.asList(1, 2), cache1);
            assertEquals(Arrays.asList(3, 4, 5), cache2);
        }
    }
}
