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

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.aggregation.Aggregations;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This test shows how to convert dataSet.groupBy(...).aggregate(...).aggregate(...) into
 * dataStream.keyBy(...).window(...).reduce(...).
 */
public class DataSetAggregateTest extends AbstractTestBase {
    List<Tuple3<String, Integer, Integer>> input;
    List<Tuple3<String, Integer, Integer>> expectedOutput;

    @Before
    public void before() {
        input = new ArrayList<>();
        input.add(new Tuple3<>("a", 1, 2));
        input.add(new Tuple3<>("a", 2, 4));
        input.add(new Tuple3<>("b", 3, 6));
        input.add(new Tuple3<>("b", 4, 8));
        input.add(new Tuple3<>("c", 5, 10));
        input.add(new Tuple3<>("c", 6, 12));
        Collections.shuffle(input);

        expectedOutput = new ArrayList<>();
        expectedOutput.add(new Tuple3<>("a", 3, 4));
        expectedOutput.add(new Tuple3<>("b", 7, 8));
        expectedOutput.add(new Tuple3<>("c", 11, 12));
    }

    @Test
    // This is an example program that uses dataSet.groupBy(...).aggregate(...).aggregate(...).
    public void testDataSetAggregate() throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataSet<Tuple3<String, Integer, Integer>> dataA = env.fromCollection(input);
        DataSet<Tuple3<String, Integer, Integer>> dataB =
                dataA.groupBy(0).aggregate(Aggregations.SUM, 1).and(Aggregations.MAX, 2);
        List<Tuple3<String, Integer, Integer>> result = dataB.collect();

        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }

    @Test
    // This program shows how to convert the above program into
    // dataStream.keyBy(...).window(...).reduce(...).
    public void testDataStreamReduce() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataStream<Tuple3<String, Integer, Integer>> dataA = env.fromCollection(input);
        ReduceFunction<Tuple3<String, Integer, Integer>> reduceFunction =
                new ReduceFunctionBuilder<Tuple3<String, Integer, Integer>>().sum(1).max(2).build();

        DataStream<Tuple3<String, Integer, Integer>> dataB =
                dataA.keyBy(tuple -> tuple.f0)
                        .window(EndOfStreamWindows.get())
                        .reduce(reduceFunction);
        List<Tuple3<String, Integer, Integer>> result =
                IteratorUtils.toList(dataB.executeAndCollect());

        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }
}
