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

package org.apache.flink.iteration.broadcast;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.TestLogger;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;

/** Tests the broadcastable operators. */
@RunWith(Parameterized.class)
public class BroadcastOutputTest extends TestLogger {

    private static final int NUM_RECORDS = 100;

    /** Whether object reuse is used. */
    private final boolean objectReuseEnabled;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {{true}, {false}});
    }

    public BroadcastOutputTest(boolean objectReuseEnabled) {
        this.objectReuseEnabled = objectReuseEnabled;
    }

    @Test
    public void testBroadcastWithChain() throws Exception {
        StreamExecutionEnvironment env = createTestEnvironment();

        env.addSource(new TestSource())
                .transform(
                        "broadcast", TypeInformation.of(Integer.class), new TestBroadcastOperator())
                .addSink(new CheckResultSink());
        env.execute();
    }

    @Test
    public void testBroadcastWithResultPartition() throws Exception {
        StreamExecutionEnvironment env = createTestEnvironment();

        env.addSource(new TestSource())
                .transform(
                        "broadcast", TypeInformation.of(Integer.class), new TestBroadcastOperator())
                .addSink(new CheckResultSink())
                .setParallelism(2);
        env.execute();
    }

    @Test
    public void testBroadcastWithMultipleChain() throws Exception {
        StreamExecutionEnvironment env = createTestEnvironment();

        DataStream<Integer> dataStream =
                env.addSource(new TestSource())
                        .transform(
                                "broadcast",
                                TypeInformation.of(Integer.class),
                                new TestBroadcastOperator());

        dataStream.addSink(new CheckResultSink());
        dataStream.addSink(new CheckResultSink());

        env.execute();
    }

    @Test
    public void testBroadcastWithMultipleResultPartitions() throws Exception {
        StreamExecutionEnvironment env = createTestEnvironment();

        DataStream<Integer> dataStream =
                env.addSource(new TestSource())
                        .transform(
                                "broadcast",
                                TypeInformation.of(Integer.class),
                                new TestBroadcastOperator());

        dataStream.addSink(new CheckResultSink()).setParallelism(2);
        dataStream.addSink(new CheckResultSink()).setParallelism(2);

        env.execute();
    }

    @Test
    public void testBroadcastWithMixedOutput() throws Exception {
        StreamExecutionEnvironment env = createTestEnvironment();

        DataStream<Integer> dataStream =
                env.addSource(new TestSource())
                        .transform(
                                "broadcast",
                                TypeInformation.of(Integer.class),
                                new TestBroadcastOperator());

        dataStream.addSink(new CheckResultSink()).setParallelism(1);
        dataStream.addSink(new CheckResultSink()).setParallelism(2);
        dataStream.addSink(new CheckResultSink()).setParallelism(4);

        env.execute();
    }

    @Test
    public void testBroadcastWithMixedOutputWithSideOutput() throws Exception {
        StreamExecutionEnvironment env = createTestEnvironment();

        SingleOutputStreamOperator<Integer> dataStream =
                env.addSource(new TestSource())
                        .transform(
                                "broadcast",
                                TypeInformation.of(Integer.class),
                                new TestBroadcastOperator());

        dataStream.addSink(new CheckResultSink());
        dataStream.getSideOutput(new OutputTag<Integer>("0") {}).addSink(new CheckResultSink());
        dataStream
                .getSideOutput(new OutputTag<Integer>("1") {})
                .addSink(new CheckResultSink())
                .setParallelism(2);
        dataStream
                .getSideOutput(new OutputTag<Integer>("2") {})
                .addSink(new CheckResultSink())
                .setParallelism(4);

        env.execute();
    }

    // ------------------------------------------------------------------------
    //  Utilities
    // ------------------------------------------------------------------------

    private StreamExecutionEnvironment createTestEnvironment() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.createLocalEnvironment();

        if (objectReuseEnabled) {
            env.getConfig().enableObjectReuse();
        } else {
            env.getConfig().disableObjectReuse();
        }
        env.getConfig().disableGenericTypes();
        env.setParallelism(1);

        return env;
    }

    /** A test source that emits a list of numbers. */
    private static class TestSource implements SourceFunction<Integer> {
        @Override
        public void run(SourceContext<Integer> ctx) throws Exception {
            for (int i = 0; i < NUM_RECORDS; ++i) {
                ctx.collect(i);
            }
        }

        @Override
        public void cancel() {}
    }

    /** The test broadcasting operator that broadcasts all the records received. */
    private static class TestBroadcastOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Integer, Integer> {

        private BroadcastOutput<Integer> broadcastOutput;

        @Override
        public void setup(
                StreamTask<?, ?> containingTask,
                StreamConfig config,
                Output<StreamRecord<Integer>> output) {
            super.setup(containingTask, config, output);
            broadcastOutput =
                    BroadcastOutputFactory.createBroadcastOutput(
                            output, metrics.getIOMetricGroup().getNumRecordsOutCounter());
        }

        @Override
        public void processElement(StreamRecord<Integer> element) throws Exception {
            broadcastOutput.broadcastEmit(element);
        }
    }

    /**
     * The test sink that checks whether all the records are received. If not, it will throw
     * exceptions and finally cause the tests to fail.
     */
    private static class CheckResultSink extends RichSinkFunction<Integer> {
        private List<Integer> received;

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            received = new ArrayList<>(NUM_RECORDS);
        }

        @Override
        public void invoke(Integer value, Context context) {
            received.add(value);
        }

        @Override
        public void close() {
            assertEquals(
                    "Number of received records does not consistent", NUM_RECORDS, received.size());
            for (int i = 0; i < NUM_RECORDS; ++i) {
                assertEquals(
                        String.format("The %d elements does not consistent", i),
                        received.get(i),
                        Integer.valueOf(i));
            }
        }
    }
}
