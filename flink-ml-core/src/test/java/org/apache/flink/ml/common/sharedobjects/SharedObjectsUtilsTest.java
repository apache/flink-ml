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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.LongSerializer;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Tests the {@link SharedObjectsUtils}. */
public class SharedObjectsUtilsTest {

    private static final ItemDescriptor<Long> SUM =
            ItemDescriptor.of("sum", LongSerializer.INSTANCE, 0L);
    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    static SharedObjectsBody.SharedObjectsBodyResult sharedObjectsBody(List<DataStream<?>> inputs) {
        //noinspection unchecked
        DataStream<Long> data = (DataStream<Long>) inputs.get(0);

        AOperator aOp = new AOperator();
        SingleOutputStreamOperator<Long> afterAOp =
                data.transform("a", TypeInformation.of(Long.class), aOp);

        BOperator bOp = new BOperator();
        SingleOutputStreamOperator<Long> afterBOp =
                afterAOp.transform("b", TypeInformation.of(Long.class), bOp);

        Map<ItemDescriptor<?>, SharedObjectsStreamOperator> ownerMap = new HashMap<>();
        ownerMap.put(SUM, aOp);

        return new SharedObjectsBody.SharedObjectsBodyResult(
                Collections.singletonList(afterBOp),
                Arrays.asList(afterAOp.getTransformation(), afterBOp.getTransformation()),
                ownerMap);
    }

    @Test
    public void testSharedObjects() throws Exception {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();

        DataStream<Long> data = env.fromSequence(1, 100);
        List<DataStream<?>> outputs =
                SharedObjectsUtils.withSharedObjects(
                        Collections.singletonList(data), SharedObjectsUtilsTest::sharedObjectsBody);
        //noinspection unchecked
        DataStream<Long> partitionSum = (DataStream<Long>) outputs.get(0);
        DataStream<Long> allSum = DataStreamUtils.reduce(partitionSum, new SumReduceFunction());
        allSum.getTransformation().setParallelism(1);
        //noinspection unchecked
        List<Long> results = IteratorUtils.toList(allSum.executeAndCollect());
        Assert.assertEquals(Collections.singletonList(5050L), results);
    }

    /** Operator A: add input elements to the shared {@link #SUM}. */
    static class AOperator extends AbstractStreamOperator<Long>
            implements OneInputStreamOperator<Long, Long>,
                    SharedObjectsStreamOperator,
                    BoundedOneInput {

        private final String sharedObjectsAccessorID;
        private SharedObjectsContext sharedObjectsContext;

        public AOperator() {
            sharedObjectsAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
        }

        @Override
        public void onSharedObjectsContextSet(SharedObjectsContext context) {
            this.sharedObjectsContext = context;
        }

        @Override
        public String getSharedObjectsAccessorID() {
            return sharedObjectsAccessorID;
        }

        @Override
        public void processElement(StreamRecord<Long> element) throws Exception {
            sharedObjectsContext.invoke(
                    (getter, setter) -> {
                        Long currentSum = getter.get(SUM);
                        setter.set(SUM, currentSum + element.getValue());
                    });
        }

        @Override
        public void endInput() throws Exception {
            // Informs BOperator to get the value from shared {@link #SUM}.
            output.collect(new StreamRecord<>(0L));
        }
    }

    /** Operator B: when input ends, get the value from shared {@link #SUM}. */
    static class BOperator extends AbstractStreamOperator<Long>
            implements OneInputStreamOperator<Long, Long>, SharedObjectsStreamOperator {

        private final String sharedObjectsAccessorID;
        private SharedObjectsContext sharedObjectsContext;

        public BOperator() {
            sharedObjectsAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
        }

        @Override
        public void onSharedObjectsContextSet(SharedObjectsContext context) {
            this.sharedObjectsContext = context;
        }

        @Override
        public String getSharedObjectsAccessorID() {
            return sharedObjectsAccessorID;
        }

        @Override
        public void processElement(StreamRecord<Long> element) throws Exception {
            sharedObjectsContext.invoke(
                    (getter, setter) -> {
                        output.collect(new StreamRecord<>(getter.get(SUM)));
                    });
        }
    }

    static class SumReduceFunction implements ReduceFunction<Long> {
        @Override
        public Long reduce(Long value1, Long value2) {
            return value1 + value2;
        }
    }
}
