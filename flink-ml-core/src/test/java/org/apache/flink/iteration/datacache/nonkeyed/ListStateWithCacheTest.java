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
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.MemorySize;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.configuration.TaskManagerOptions;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.memory.MemoryReservationException;
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

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.testcontainers.shaded.org.apache.commons.lang3.RandomStringUtils;

/** Tests {@link ListStateWithCache}. */
public class ListStateWithCacheTest {

    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

    private MiniClusterConfiguration createMiniClusterConfiguration() {
        Configuration configuration = new Configuration();
        // Set managed memory size to a small value, so when the instance of ListStateWithCache
        // tries to allocate memory, it will use up all managed memory assigned to itself.
        configuration.set(TaskManagerOptions.MANAGED_MEMORY_SIZE, MemorySize.ofMebiBytes(16));
        configuration.set(RestOptions.PORT, 18082);
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        return new MiniClusterConfiguration.Builder()
                .setConfiguration(configuration)
                .setNumTaskManagers(1)
                .setNumSlotsPerTaskManager(1)
                .build();
    }

    @Test
    public void testCorrectMemorySubFraction() throws Exception {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration())) {
            miniCluster.start();
            JobGraph jobGraph = getJobGraph(2, 1. / 2);
            miniCluster.executeJobBlocking(jobGraph);
        }
    }

    @Test
    public void testIncorrectMemorySubFraction() {
        try (MiniCluster miniCluster = new MiniCluster(createMiniClusterConfiguration())) {
            miniCluster.start();
            JobGraph jobGraph = getJobGraph(2, 0.6);
            miniCluster.executeJobBlocking(jobGraph);
        } catch (Exception e) {
            Throwable rootCause = ExceptionUtils.getRootCause(e);
            Assert.assertEquals(MemoryReservationException.class, rootCause.getClass());
        }
    }

    private JobGraph getJobGraph(int times, double memorySubFraction) {
        Configuration configuration = new Configuration();
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment(configuration);
        env.setParallelism(1);

        final int n = 10;
        DataStream<String> data =
                env.fromSequence(1, n).map(d -> RandomStringUtils.randomAlphabetic(1024 * 1024));
        DataStream<Integer> counter =
                data.transform("cache", Types.INT, new CacheDataOperator(times, memorySubFraction));
        DataStreamUtils.setManagedMemoryWeight(counter, 100);
        counter.addSink(
                new SinkFunction<Integer>() {
                    @Override
                    public void invoke(Integer value, Context context) throws Exception {
                        Assert.assertEquals((Integer) (n * times), value);
                        SinkFunction.super.invoke(value, context);
                    }
                });
        return env.getStreamGraph().getJobGraph();
    }

    private static class CacheDataOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<String, Integer>,
                    BoundedOneInput,
                    StreamOperatorStateHandler.CheckpointedStreamOperator {

        private final int times;
        private final double memorySubFraction;
        private transient ListStateWithCache<String>[] cached;

        public CacheDataOperator(int times, double memorySubFraction) {
            this.times = times;
            this.memorySubFraction = memorySubFraction;
        }

        @Override
        public void processElement(StreamRecord<String> element) throws Exception {
            for (int i = 0; i < times; i += 1) {
                cached[i].add(element.getValue());
            }
        }

        @Override
        public void endInput() throws Exception {
            int counter = 0;
            for (int i = 0; i < times; i += 1) {
                for (String ignored : cached[i].get()) {
                    counter += 1;
                }
                cached[i].clear();
            }
            output.collect(new StreamRecord<>(counter));
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            //noinspection unchecked
            cached = new ListStateWithCache[times];
            for (int i = 0; i < times; i += 1) {
                cached[i] =
                        new ListStateWithCache<>(
                                StringSerializer.INSTANCE,
                                getContainingTask(),
                                getRuntimeContext(),
                                context,
                                getOperatorID(),
                                memorySubFraction);
            }
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
        }
    }
}
