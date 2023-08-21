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

package org.apache.flink.iteration.datacache.nonkeyed;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.MemorySize;
import org.apache.flink.configuration.TaskManagerOptions;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorStateHandler;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.apache.commons.lang3.RandomStringUtils;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Random;

/** Tests {@link ListStateWithCache}. */
public class ListStateWithCacheTest {

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    private MiniClusterConfiguration createMiniClusterConfiguration() {
        Configuration configuration = new Configuration();
        // Set managed memory size to a small value, so when the instance of ListStateWithCache
        // tries to allocate memory, it will use up all managed memory assigned to itself.
        configuration.set(TaskManagerOptions.MANAGED_MEMORY_SIZE, MemorySize.ofMebiBytes(16));
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        return new MiniClusterConfiguration.Builder()
                .setConfiguration(configuration)
                .setNumTaskManagers(1)
                .setNumSlotsPerTaskManager(1)
                .build();
    }

    @Test
    public void testWithMemoryWeights() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration())) {
            miniCluster.start();
            int n = 5;
            Random random = new Random();
            double[] weights = new double[n];
            for (int i = 0; i < n; i += 1) {
                weights[i] = random.nextInt(100);
            }
            JobGraph jobGraph = getJobGraph(weights);
            miniCluster.executeJobBlocking(jobGraph);
        }
    }

    private JobGraph getJobGraph(double[] weights) {
        Configuration configuration = new Configuration();
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment(configuration);
        env.setParallelism(1);

        final int n = 10;
        DataStream<String> data =
                env.fromSequence(1, n).map(d -> RandomStringUtils.randomAlphabetic(1024 * 1024));
        DataStream<Integer> counter =
                data.transform("cache", Types.INT, new CacheDataOperator(weights));
        DataStreamUtils.setManagedMemoryWeight(counter, 100);
        counter.addSink(
                new SinkFunction<Integer>() {
                    @Override
                    public void invoke(Integer value, Context context) throws Exception {
                        Assert.assertEquals((Integer) (n * weights.length), value);
                        SinkFunction.super.invoke(value, context);
                    }
                });
        return env.getStreamGraph().getJobGraph();
    }

    private static class CacheDataOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<String, Integer>,
                    BoundedOneInput,
                    StreamOperatorStateHandler.CheckpointedStreamOperator {
        private final double[] weights;
        private transient ListStateWithCache<String>[] cached;

        public CacheDataOperator(double[] weights) {
            this.weights = weights;
        }

        @Override
        public void processElement(StreamRecord<String> element) throws Exception {
            for (int i = 0; i < weights.length; i += 1) {
                cached[i].add(element.getValue());
            }
        }

        @Override
        public void endInput() throws Exception {
            int counter = 0;
            for (int i = 0; i < weights.length; i += 1) {
                for (String ignored : cached[i].get()) {
                    counter += 1;
                }
                cached[i].clear();
            }
            output.collect(new StreamRecord<>(counter));
        }

        @SuppressWarnings("unchecked")
        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            final OperatorID operatorID = getOperatorID();
            TypeSerializer<String>[] serializers =
                    (TypeSerializer<String>[]) new TypeSerializer[weights.length];
            final OperatorScopeManagedMemoryManager manager =
                    OperatorScopeManagedMemoryManager.getOrCreate(operatorID);
            for (int i = 0; i < weights.length; i += 1) {
                serializers[i] = StringSerializer.INSTANCE;
                manager.register("state-" + i, weights[i]);
            }
            cached = new ListStateWithCache[weights.length];
            for (int i = 0; i < weights.length; i += 1) {
                cached[i] =
                        new ListStateWithCache<>(
                                serializers[i],
                                "state-" + i,
                                getContainingTask(),
                                getRuntimeContext(),
                                context,
                                operatorID);
            }
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
        }
    }
}
