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

package org.apache.flink.ml.common.datastream;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;
import org.apache.flink.util.NumberSequenceIterator;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;

/** Tests the {@link DataStreamUtils}. */
public class DataStreamUtilsTest {
    private StreamExecutionEnvironment env;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
    }

    @Test
    public void testMapPartition() throws Exception {
        DataStream<Long> dataStream =
                env.fromParallelCollection(new NumberSequenceIterator(0L, 19L), Types.LONG);
        DataStream<Integer> countsPerPartition =
                DataStreamUtils.mapPartition(dataStream, new TestMapPartitionFunc());
        List<Integer> counts = IteratorUtils.toList(countsPerPartition.executeAndCollect());
        assertArrayEquals(
                new int[] {5, 5, 5, 5}, counts.stream().mapToInt(Integer::intValue).toArray());
    }

    /** A simple implementation for a {@link MapPartitionFunction}. */
    private static class TestMapPartitionFunc extends RichMapPartitionFunction<Long, Integer> {

        public void mapPartition(Iterable<Long> values, Collector<Integer> out) {
            assertNotNull(getRuntimeContext());
            int cnt = 0;
            for (long ignored : values) {
                cnt++;
            }
            out.collect(cnt);
        }
    }
}
