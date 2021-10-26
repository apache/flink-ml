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

import org.apache.flink.api.common.functions.JoinFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * This test shows how to convert
 * dataSetA.join(datasetB).where(...).equalTo(...).projectFirst(...).projectSecond(...) into
 * dataStreamA.join(dataStreamB).where(...).equalTo(...).window(...).apply(...).
 *
 * <p>Reference: https://issues.apache.org/jira/browse/FLINK-22587
 */
public class DataSetJoinTest extends AbstractTestBase {
    List<Tuple2<String, Integer>> inputA;
    List<Tuple2<String, Integer>> inputB;

    List<Tuple3<String, Integer, Integer>> expectedOutput;

    @Before
    public void before() {
        inputA = new ArrayList<>();
        inputA.add(new Tuple2<>("a", 1));
        inputA.add(new Tuple2<>("a", 2));
        inputA.add(new Tuple2<>("b", 3));
        inputA.add(new Tuple2<>("c", 4));

        inputB = new ArrayList<>();
        inputB.add(new Tuple2<>("a", 5));
        inputB.add(new Tuple2<>("a", 6));
        inputB.add(new Tuple2<>("b", 7));
        inputB.add(new Tuple2<>("d", 8));

        expectedOutput = new ArrayList<>();
        expectedOutput.add(new Tuple3<>("a", 1, 5));
        expectedOutput.add(new Tuple3<>("a", 1, 6));
        expectedOutput.add(new Tuple3<>("a", 2, 5));
        expectedOutput.add(new Tuple3<>("a", 2, 6));
        expectedOutput.add(new Tuple3<>("b", 3, 7));
    }

    @Test
    // This is an example program that uses
    // dataSetA.join(datasetB).where(...).equalTo(...).projectFirst(...).projectSecond(...).
    public void testDataSetJoin() throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataSet<Tuple2<String, Integer>> dataA = env.fromCollection(inputA);
        DataSet<Tuple2<String, Integer>> dataB = env.fromCollection(inputB);
        DataSet<Tuple3<String, Integer, Integer>> dataC =
                dataA.join(dataB).where(0).equalTo(0).projectFirst(0, 1).projectSecond(1);

        List<Tuple3<String, Integer, Integer>> result = dataC.collect();

        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }

    @Test
    // This program shows how to convert the above program into
    // dataStreamA.join(dataStreamB).where(...).equalTo(...).window(...).apply(...).
    public void testDataStreamJoin() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataStream<Tuple2<String, Integer>> dataA = env.fromCollection(inputA);
        DataStream<Tuple2<String, Integer>> dataB = env.fromCollection(inputB);
        DataStream<Tuple3<String, Integer, Integer>> dataC =
                dataA.join(dataB)
                        .where(tuple -> tuple.f0)
                        .equalTo(tuple -> tuple.f0)
                        .window(EndOfStreamWindows.get())
                        .apply(new MyJoinFunction());

        List<Tuple3<String, Integer, Integer>> result =
                IteratorUtils.toList(dataC.executeAndCollect());
        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }

    private static class MyJoinFunction
            implements JoinFunction<
                    Tuple2<String, Integer>,
                    Tuple2<String, Integer>,
                    Tuple3<String, Integer, Integer>> {
        @Override
        public Tuple3<String, Integer, Integer> join(
                Tuple2<String, Integer> v1, Tuple2<String, Integer> v2) throws Exception {
            return Tuple3.of(v1.f0, v1.f1, v2.f1);
        }
    }
}
