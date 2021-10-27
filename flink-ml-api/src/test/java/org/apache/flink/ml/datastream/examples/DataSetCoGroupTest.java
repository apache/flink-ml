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

package org.apache.flink.ml.datastream.examples;

import org.apache.flink.api.common.functions.CoGroupFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.EndOfStreamWindows;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.util.Collector;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * This test shows how to convert dataSetA.coGroup(datasetB).where(...).equalTo(...).with(...) into
 * dataStreamA.coGroup(dataStreamB).where(...).equalTo(...).window(...).apply(...).
 */
public class DataSetCoGroupTest extends AbstractTestBase {
    List<Tuple2<String, Integer>> inputA;
    List<Tuple2<String, Integer>> inputB;

    List<Tuple2<String, Integer>> expectedOutput;

    @Before
    public void before() {
        inputA = new ArrayList<>();
        inputA.add(new Tuple2<>("a", 1));
        inputA.add(new Tuple2<>("a", 2));
        inputA.add(new Tuple2<>("b", 2));
        inputA.add(new Tuple2<>("b", 4));
        inputA.add(new Tuple2<>("c", 3));
        inputA.add(new Tuple2<>("c", 6));

        inputB = new ArrayList<>();
        inputB.add(new Tuple2<>("a", 4));
        inputB.add(new Tuple2<>("b", 5));
        inputB.add(new Tuple2<>("d", 6));

        expectedOutput = new ArrayList<>();
        expectedOutput.add(new Tuple2<>("a", 7));
        expectedOutput.add(new Tuple2<>("b", 11));
        expectedOutput.add(new Tuple2<>("c", 9));
        expectedOutput.add(new Tuple2<>("d", 6));
    }

    @Test
    // This is an example program that uses
    // dataSetA.coGroup(datasetB).where(...).equalTo(...).with(...).
    public void testDataSetCoGroup() throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataSet<Tuple2<String, Integer>> dataA = env.fromCollection(inputA);
        DataSet<Tuple2<String, Integer>> dataB = env.fromCollection(inputB);
        DataSet<Tuple2<String, Integer>> dataC =
                dataA.coGroup(dataB).where(0).equalTo(0).with(new MyCoGroupFunction());

        List<Tuple2<String, Integer>> result = dataC.collect();
        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }

    private static class MyCoGroupFunction
            implements CoGroupFunction<
                    Tuple2<String, Integer>, Tuple2<String, Integer>, Tuple2<String, Integer>> {
        @Override
        public void coGroup(
                Iterable<Tuple2<String, Integer>> left,
                Iterable<Tuple2<String, Integer>> right,
                Collector<Tuple2<String, Integer>> out) {
            String key = null;
            int sum = 0;

            for (Tuple2<String, Integer> value : left) {
                if (key == null) {
                    key = value.f0;
                }
                Assert.assertEquals(key, value.f0);
                sum += value.f1;
            }
            for (Tuple2<String, Integer> value : right) {
                if (key == null) {
                    key = value.f0;
                }
                Assert.assertEquals(key, value.f0);
                sum += value.f1;
            }

            out.collect(new Tuple2<>(key, sum));
        }
    }

    @Test
    // This program shows how to convert the above program into
    // dataStreamA.coGroup(dataStreamB).where(...).equalTo(...).window(...).apply(...).
    public void testDataStreamCoGroup() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataStream<Tuple2<String, Integer>> dataA = env.fromCollection(inputA);
        DataStream<Tuple2<String, Integer>> dataB = env.fromCollection(inputB);
        DataStream<Tuple2<String, Integer>> dataC =
                dataA.coGroup(dataB)
                        .where(tuple -> tuple.f0)
                        .equalTo(tuple -> tuple.f0)
                        .window(EndOfStreamWindows.get())
                        .apply(new MyCoGroupFunction());

        List<Tuple2<String, Integer>> result = IteratorUtils.toList(dataC.executeAndCollect());
        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }
}
