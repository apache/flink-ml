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
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.runtime.state.storage.FileSystemCheckpointStorage;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.RandomStringUtils;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Tests the {@link SharedObjectsUtils}. */
public class SharedObjectsUtilsTest {

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void testWithDataDeps() throws Exception {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();

        DataStream<Long> data = env.fromSequence(1, 100);
        List<DataStream<?>> outputs =
                SharedObjectsUtils.withSharedObjects(
                        Collections.singletonList(data), new SharedObjectsBodyWithDataDeps());
        //noinspection unchecked
        DataStream<Long> partitionSum = (DataStream<Long>) outputs.get(0);
        DataStream<Long> allSum =
                DataStreamUtils.reduce(
                        partitionSum, new SharedObjectsBodyWithDataDeps.SumReduceFunction());
        allSum.getTransformation().setParallelism(1);
        //noinspection unchecked
        List<Long> results = IteratorUtils.toList(allSum.executeAndCollect());
        Assert.assertEquals(Collections.singletonList(5050L), results);
    }

    @Test
    public void testWithoutDataDeps() throws Exception {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();

        DataStream<Long> data = env.fromSequence(1, 100);
        List<DataStream<?>> outputs =
                SharedObjectsUtils.withSharedObjects(
                        Collections.singletonList(data), new SharedObjectsBodyWithoutDataDeps());
        //noinspection unchecked
        DataStream<Long> added = (DataStream<Long>) outputs.get(0);
        //noinspection unchecked
        List<Long> results = IteratorUtils.toList(added.executeAndCollect());
        Collections.sort(results);
        List<Long> expected = new ArrayList<>();
        for (long i = 1; i <= 100; i += 1) {
            expected.add(i + 5050);
        }
        Assert.assertEquals(expected, results);
    }

    @Test
    public void testPotentialDeadlock() throws Exception {
        Configuration configuration = new Configuration();
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment(configuration);
        File stateFolder = tempFolder.newFolder();
        env.getCheckpointConfig()
                .setCheckpointStorage(
                        new FileSystemCheckpointStorage(
                                new Path("file://", stateFolder.getPath())));
        final int n = 100;
        // Set it to a large value, thus causing a deadlock situation.
        final int len = 1 << 20;
        DataStream<String> data =
                env.fromSequence(1, n).map(d -> RandomStringUtils.randomAlphabetic(len));
        List<DataStream<?>> outputs =
                SharedObjectsUtils.withSharedObjects(
                        Collections.singletonList(data), new SharedObjectsBodyPotentialDeadlock());
        //noinspection unchecked
        DataStream<String> added = (DataStream<String>) outputs.get(0);
        added.addSink(
                new SinkFunction<String>() {
                    @Override
                    public void invoke(String value, Context context) {
                        Assert.assertEquals(2 * len, value.length());
                    }
                });
        env.execute();
    }

    static class SharedObjectsBodyWithDataDeps implements SharedObjectsBody {
        private static final Descriptor<Long> SUM =
                Descriptor.of("sum", LongSerializer.INSTANCE, 0L);

        @Override
        public SharedObjectsBodyResult process(List<DataStream<?>> inputs) {
            //noinspection unchecked
            DataStream<Long> data = (DataStream<Long>) inputs.get(0);

            AOperator aOp = new AOperator();
            SingleOutputStreamOperator<Long> afterAOp =
                    data.transform("a", TypeInformation.of(Long.class), aOp);

            BOperator bOp = new BOperator();
            SingleOutputStreamOperator<Long> afterBOp =
                    afterAOp.transform("b", TypeInformation.of(Long.class), bOp);

            Map<Descriptor<?>, AbstractSharedObjectsStreamOperator<?>> ownerMap = new HashMap<>();
            ownerMap.put(SUM, aOp);

            return new SharedObjectsBodyResult(
                    Collections.singletonList(afterBOp),
                    Arrays.asList(afterAOp.getTransformation(), afterBOp.getTransformation()),
                    ownerMap);
        }

        /** Operator A: add input elements to the shared {@link #SUM}. */
        static class AOperator extends AbstractSharedObjectsOneInputStreamOperator<Long, Long>
                implements BoundedOneInput {

            private transient long sum = 0;

            @Override
            public void processElement(StreamRecord<Long> element) throws Exception {
                sum += element.getValue();
            }

            @Override
            public void endInput() throws Exception {
                context.write(SUM, sum);
                // Informs BOperator to get the value from shared {@link #SUM}.
                output.collect(new StreamRecord<>(0L));
            }

            @Override
            public List<ReadRequest<?>> readRequestsInProcessElement() {
                return Collections.emptyList();
            }
        }

        /** Operator B: when input ends, get the value from shared {@link #SUM}. */
        static class BOperator extends AbstractSharedObjectsOneInputStreamOperator<Long, Long> {

            @Override
            public void processElement(StreamRecord<Long> element) throws Exception {
                output.collect(new StreamRecord<>(context.read(SUM.sameStep())));
            }

            @Override
            public List<ReadRequest<?>> readRequestsInProcessElement() {
                return Collections.singletonList(SUM.sameStep());
            }
        }

        static class SumReduceFunction implements ReduceFunction<Long> {
            @Override
            public Long reduce(Long value1, Long value2) {
                return value1 + value2;
            }
        }
    }

    static class SharedObjectsBodyWithoutDataDeps implements SharedObjectsBody {
        private static final Descriptor<Long> SUM = Descriptor.of("sum", LongSerializer.INSTANCE);

        @Override
        public SharedObjectsBodyResult process(List<DataStream<?>> inputs) {
            //noinspection unchecked
            DataStream<Long> data = (DataStream<Long>) inputs.get(0);
            DataStream<Long> sum = DataStreamUtils.reduce(data, Long::sum);

            COperator cOp = new COperator();
            SingleOutputStreamOperator<Long> afterCOp =
                    sum.broadcast().transform("c", TypeInformation.of(Long.class), cOp);

            DOperator dOp = new DOperator();
            SingleOutputStreamOperator<Long> afterDOp =
                    data.transform("d", TypeInformation.of(Long.class), dOp);

            Map<Descriptor<?>, AbstractSharedObjectsStreamOperator<?>> ownerMap = new HashMap<>();
            ownerMap.put(SUM, cOp);

            return new SharedObjectsBodyResult(
                    Collections.singletonList(afterDOp),
                    Arrays.asList(afterCOp.getTransformation(), afterDOp.getTransformation()),
                    ownerMap);
        }

        /** Operator C: set the shared object. */
        static class COperator extends AbstractSharedObjectsOneInputStreamOperator<Long, Long>
                implements BoundedOneInput {
            private transient long sum;

            @Override
            public void processElement(StreamRecord<Long> element) throws Exception {
                sum = element.getValue();
            }

            @Override
            public void endInput() throws Exception {
                Thread.sleep(2 * 1000);
                context.write(SUM, sum);
            }

            @Override
            public List<ReadRequest<?>> readRequestsInProcessElement() {
                return Collections.emptyList();
            }
        }

        /** Operator D: get the value from shared {@link #SUM}. */
        static class DOperator extends AbstractSharedObjectsOneInputStreamOperator<Long, Long> {

            private Long sum;

            @Override
            public void processElement(StreamRecord<Long> element) throws Exception {
                if (null == sum) {
                    sum = context.read(SUM.sameStep());
                }
                output.collect(new StreamRecord<>(sum + element.getValue()));
            }

            @Override
            public List<ReadRequest<?>> readRequestsInProcessElement() {
                return Collections.singletonList(SUM.sameStep());
            }
        }
    }

    static class SharedObjectsBodyPotentialDeadlock implements SharedObjectsBody {
        private static final Descriptor<String> LAST =
                Descriptor.of("last", StringSerializer.INSTANCE);

        @Override
        public SharedObjectsBodyResult process(List<DataStream<?>> inputs) {
            //noinspection unchecked
            DataStream<String> data = (DataStream<String>) inputs.get(0);
            DataStream<String> sum = DataStreamUtils.reduce(data, (v1, v2) -> v2);

            EOperator eOp = new EOperator();
            SingleOutputStreamOperator<String> afterCOp =
                    sum.broadcast().transform("e", TypeInformation.of(String.class), eOp);

            FOperator dOp = new FOperator();
            SingleOutputStreamOperator<String> afterDOp =
                    data.transform("d", TypeInformation.of(String.class), dOp);

            Map<Descriptor<?>, AbstractSharedObjectsStreamOperator<?>> ownerMap = new HashMap<>();
            ownerMap.put(LAST, eOp);

            return new SharedObjectsBodyResult(
                    Collections.singletonList(afterDOp),
                    Arrays.asList(afterCOp.getTransformation(), afterDOp.getTransformation()),
                    ownerMap);
        }

        /** Operator E: set the shared object. */
        static class EOperator extends AbstractSharedObjectsOneInputStreamOperator<String, String>
                implements BoundedOneInput {
            private transient String last;

            @Override
            public void processElement(StreamRecord<String> element) throws Exception {
                last = element.getValue();
            }

            @Override
            public void endInput() throws Exception {
                Thread.sleep(2 * 1000);
                context.write(LAST, last);
            }

            @Override
            public List<ReadRequest<?>> readRequestsInProcessElement() {
                return Collections.emptyList();
            }
        }

        /** Operator F: get the value from shared {@link #LAST}. */
        static class FOperator extends AbstractSharedObjectsOneInputStreamOperator<String, String> {

            private String last;

            @Override
            public void processElement(StreamRecord<String> element) throws Exception {
                if (null == last) {
                    last = context.read(LAST.sameStep());
                }
                output.collect(new StreamRecord<>(last + element.getValue()));
            }

            @Override
            public List<ReadRequest<?>> readRequestsInProcessElement() {
                return Collections.singletonList(LAST.sameStep());
            }
        }
    }
}
