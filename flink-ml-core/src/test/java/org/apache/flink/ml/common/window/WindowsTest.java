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

package org.apache.flink.ml.common.window;

import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.ProcessAllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.Window;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.util.Collector;

import org.apache.commons.collections.IteratorUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Tests the {@link Windows}. */
@SuppressWarnings("unchecked")
public class WindowsTest extends AbstractTestBase {
    private static final int RECORD_NUM = 100;

    private static List<Long> inputData;

    private static DataStream<Long> inputStream;
    private static DataStream<Long> inputStreamWithProcessingTimeGap;
    private static DataStream<Long> inputStreamWithEventTime;

    @BeforeClass
    public static void beforeClass() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        inputData = new ArrayList<>();
        for (long i = 0; i < RECORD_NUM; i++) {
            inputData.add(i);
        }
        inputStream = env.fromCollection(inputData);

        inputStreamWithProcessingTimeGap =
                inputStream
                        .map(
                                new MapFunction<Long, Long>() {
                                    private int count = 0;

                                    @Override
                                    public Long map(Long value) throws Exception {
                                        count++;
                                        if (count % (RECORD_NUM / 2) == 0) {
                                            Thread.sleep(1000);
                                        }
                                        return value;
                                    }
                                })
                        .setParallelism(1);

        inputStreamWithEventTime =
                inputStream.assignTimestampsAndWatermarks(
                        WatermarkStrategy.<Long>forMonotonousTimestamps()
                                .withTimestampAssigner(
                                        (SerializableTimestampAssigner<Long>)
                                                (element, recordTimestamp) -> element));
    }

    @Test
    public void testGlobalWindows() throws Exception {
        DataStream<List<Long>> outputStream =
                DataStreamUtils.windowAllAndProcess(
                        inputStream,
                        GlobalWindows.getInstance(),
                        new CreateAllWindowBatchFunction<>(),
                        Types.LIST(Types.LONG));
        List<List<Long>> actualBatches = IteratorUtils.toList(outputStream.executeAndCollect());
        assertEquals(1, actualBatches.size());
        assertEquals(new HashSet<>(inputData), new HashSet<>(actualBatches.get(0)));
    }

    @Test
    public void testCountTumblingWindows() throws Exception {
        DataStream<List<Long>> outputStream =
                DataStreamUtils.windowAllAndProcess(
                        inputStream,
                        CountTumblingWindows.of(RECORD_NUM / 7),
                        new CreateAllWindowBatchFunction<>(),
                        Types.LIST(Types.LONG));
        List<List<Long>> actualBatches = IteratorUtils.toList(outputStream.executeAndCollect());
        assertEquals(7, actualBatches.size());
        int count = 0;
        for (List<Long> batch : actualBatches) {
            count += batch.size();
        }
        assertEquals(RECORD_NUM - (RECORD_NUM % 7), count);
    }

    @Test
    public void testProcessingTimeTumblingWindows() throws Exception {
        DataStream<List<Long>> outputStream =
                DataStreamUtils.windowAllAndProcess(
                        inputStreamWithProcessingTimeGap,
                        ProcessingTimeTumblingWindows.of(Time.milliseconds(100)),
                        new CreateAllWindowBatchFunction<>(),
                        Types.LIST(Types.LONG));
        List<List<Long>> actualBatches = IteratorUtils.toList(outputStream.executeAndCollect());
        assertTrue(actualBatches.size() > 1);
        List<Long> mergedBatches = new ArrayList<>();
        for (List<Long> batch : actualBatches) {
            mergedBatches.addAll(batch);
        }
        assertTrue(mergedBatches.containsAll(inputData.subList(0, RECORD_NUM - 1)));
    }

    @Test
    public void testEventTimeTumblingWindows() throws Exception {
        DataStream<List<Long>> outputStream =
                DataStreamUtils.windowAllAndProcess(
                        inputStreamWithEventTime,
                        EventTimeTumblingWindows.of(Time.milliseconds(RECORD_NUM / 7)),
                        new CreateAllWindowBatchFunction<>(),
                        Types.LIST(Types.LONG));
        List<List<Long>> actualBatches = IteratorUtils.toList(outputStream.executeAndCollect());
        assertEquals(8, actualBatches.size());
        List<Long> mergedBatches = new ArrayList<>();
        for (List<Long> batch : actualBatches) {
            mergedBatches.addAll(batch);
        }
        assertEquals(RECORD_NUM, mergedBatches.size());
        assertEquals(new HashSet<>(inputData), new HashSet<>(mergedBatches));
    }

    @Test
    public void testProcessingTimeSessionWindows() throws Exception {
        DataStream<List<Long>> outputStream =
                DataStreamUtils.windowAllAndProcess(
                        inputStreamWithProcessingTimeGap,
                        ProcessingTimeSessionWindows.withGap(Time.milliseconds(100)),
                        new CreateAllWindowBatchFunction<>(),
                        Types.LIST(Types.LONG));
        List<List<Long>> actualBatches = IteratorUtils.toList(outputStream.executeAndCollect());
        assertTrue(actualBatches.size() > 1);
        List<Long> mergedBatches = new ArrayList<>();
        for (List<Long> batch : actualBatches) {
            mergedBatches.addAll(batch);
        }
        assertTrue(mergedBatches.containsAll(inputData.subList(0, RECORD_NUM - 1)));
    }

    @Test
    public void testEventTimeSessionWindows() throws Exception {
        DataStream<List<Long>> outputStream =
                DataStreamUtils.windowAllAndProcess(
                        inputStreamWithEventTime,
                        EventTimeSessionWindows.withGap(Time.milliseconds(RECORD_NUM / 7)),
                        new CreateAllWindowBatchFunction<>(),
                        Types.LIST(Types.LONG));
        List<List<Long>> actualBatches = IteratorUtils.toList(outputStream.executeAndCollect());
        assertEquals(1, actualBatches.size());
        assertEquals(new HashSet<>(inputData), new HashSet<>(actualBatches.get(0)));
    }

    private static class CreateAllWindowBatchFunction<IN, W extends Window>
            extends ProcessAllWindowFunction<IN, List<IN>, W> {
        @Override
        public void process(
                ProcessAllWindowFunction<IN, List<IN>, W>.Context context,
                Iterable<IN> elements,
                Collector<List<IN>> out) {
            List<IN> list = new ArrayList<>();
            elements.forEach(list::add);
            out.collect(list);
        }
    }
}
