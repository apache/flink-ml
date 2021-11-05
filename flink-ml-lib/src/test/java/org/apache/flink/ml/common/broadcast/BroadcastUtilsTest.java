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

package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.common.functions.AbstractRichFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.ml.common.broadcast.operator.TestOneInputOp;
import org.apache.flink.ml.common.broadcast.operator.TestTwoInputOp;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/** Tests the {@link BroadcastUtils}. */
public class BroadcastUtilsTest {

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    private static final int NUM_RECORDS_PER_PARTITION = 10;

    private static final int NUM_TM = 2;

    private static final int NUM_SLOT = 2;

    private static final String[] BROADCAST_NAMES = new String[] {"source1", "source2"};

    private static final List<Integer> BROADCAST_INPUT =
            IntStream.range(0, NUM_TM * NUM_SLOT * NUM_RECORDS_PER_PARTITION)
                    .boxed()
                    .collect(Collectors.toList());

    private MiniClusterConfiguration createMiniClusterConfiguration() throws IOException {
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18082);
        configuration.set(
                IterationOptions.DATA_CACHE_PATH,
                "file://" + tempFolder.newFolder().getAbsolutePath());
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        return new MiniClusterConfiguration.Builder()
                .setConfiguration(configuration)
                .setNumTaskManagers(NUM_TM)
                .setNumSlotsPerTaskManager(NUM_SLOT)
                .build();
    }

    @Test
    public void testOneInputGraph() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration())) {
            miniCluster.start();
            JobGraph jobGraph = getJobGraph(1);
            miniCluster.executeJobBlocking(jobGraph);
        }
    }

    @Test
    public void testTwoInputGraph() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration())) {
            miniCluster.start();
            JobGraph jobGraph = getJobGraph(2);
            miniCluster.executeJobBlocking(jobGraph);
        }
    }

    private JobGraph getJobGraph(int numNonBroadcastInputs) {
        StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment(
                        new Configuration() {
                            {
                                this.set(
                                        ExecutionCheckpointingOptions
                                                .ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH,
                                        true);
                            }
                        });
        env.enableCheckpointing(500, CheckpointingMode.EXACTLY_ONCE);
        env.setParallelism(NUM_SLOT * NUM_TM);

        DataStream<Integer> source1 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        DataStream<Integer> source2 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        HashMap<String, DataStream<?>> bcStreamsMap = new HashMap<>();
        bcStreamsMap.put(BROADCAST_NAMES[0], source1);
        bcStreamsMap.put(BROADCAST_NAMES[1], source2);

        List<DataStream<?>> inputList = new ArrayList<>(1);
        // create a deadlock.
        inputList.add(source1);
        for (int i = 0; i < numNonBroadcastInputs - 1; i++) {
            inputList.add(env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION)));
        }

        Function<List<DataStream<?>>, DataStream<Integer>> func = getFunc(numNonBroadcastInputs);

        DataStream<Integer> result =
                BroadcastUtils.withBroadcastStream(inputList, bcStreamsMap, func);

        List<Integer> expectedNumSequence =
                new ArrayList<>(
                        NUM_TM * NUM_SLOT * NUM_RECORDS_PER_PARTITION * numNonBroadcastInputs);
        for (int i = 0; i < NUM_TM * NUM_SLOT * NUM_RECORDS_PER_PARTITION; i++) {
            for (int j = 0; j < numNonBroadcastInputs; j++) {
                expectedNumSequence.add(i);
            }
        }
        result.addSink(new TestSink(expectedNumSequence)).setParallelism(1);

        return env.getStreamGraph().getJobGraph();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Function<List<DataStream<?>>, DataStream<Integer>> getFunc(int numInputs) {
        if (numInputs == 1) {
            return dataStreams -> {
                DataStream input = dataStreams.get(0);
                return input.transform(
                                "one-input",
                                BasicTypeInfo.INT_TYPE_INFO,
                                new TestOneInputOp(
                                        new AbstractRichFunction() {},
                                        BROADCAST_NAMES,
                                        Arrays.asList(BROADCAST_INPUT, BROADCAST_INPUT)))
                        .name("broadcast");
            };
        } else if (numInputs == 2) {
            return dataStreams -> {
                DataStream input1 = dataStreams.get(0);
                DataStream input2 = dataStreams.get(1);
                return input1.connect(input2)
                        .transform(
                                "two-input",
                                BasicTypeInfo.INT_TYPE_INFO,
                                new TestTwoInputOp(
                                        new AbstractRichFunction() {},
                                        BROADCAST_NAMES,
                                        Arrays.asList(BROADCAST_INPUT, BROADCAST_INPUT)))
                        .name("broadcast");
            };
        }
        return null;
    }
}
