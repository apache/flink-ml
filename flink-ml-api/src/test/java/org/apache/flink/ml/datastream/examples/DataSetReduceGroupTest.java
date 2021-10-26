package org.apache.flink.ml.datastream.examples;

import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.EndOfStreamWindows;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.util.Collector;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This test shows how to convert dataSet.reduceGroup(...) into
 * dataStream.windowAll(...).apply(...).
 */
public class DataSetReduceGroupTest extends AbstractTestBase {
    List<Tuple3<String, Integer, Integer>> input;
    List<Tuple2<String, Integer>> expectedOutput;

    @Before
    public void before() {
        input = new ArrayList<>();
        input.add(new Tuple3<>("a", 1, 2));
        input.add(new Tuple3<>("a", 2, 4));
        input.add(new Tuple3<>("b", 3, 6));
        input.add(new Tuple3<>("b", 4, 8));
        input.add(new Tuple3<>("b", 5, 10));
        Collections.shuffle(input);

        expectedOutput = new ArrayList<>();
        expectedOutput.add(new Tuple2<>("a", 3));
        expectedOutput.add(new Tuple2<>("b", 12));
    }

    @Test
    // This is an example program that uses dataSet.reduceGroup(...).
    public void testDataSetReduceGroup() throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataSet<Tuple3<String, Integer, Integer>> dataA = env.fromCollection(input);
        DataSet<Tuple2<String, Integer>> dataB = dataA.reduceGroup(new MyGroupReduceFunction());
        List<Tuple2<String, Integer>> result = dataB.collect();

        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }

    private static class MyGroupReduceFunction
            implements GroupReduceFunction<
                    Tuple3<String, Integer, Integer>, Tuple2<String, Integer>> {

        @Override
        public void reduce(
                Iterable<Tuple3<String, Integer, Integer>> iterable,
                Collector<Tuple2<String, Integer>> out)
                throws Exception {
            Map<String, Integer> result = new HashMap<>();

            for (Tuple3<String, Integer, Integer> tuple : iterable) {
                int currentValue = result.getOrDefault(tuple.f0, 0);
                result.put(tuple.f0, currentValue + tuple.f1);
            }

            for (Map.Entry<String, Integer> entry : result.entrySet()) {
                out.collect(new Tuple2<>(entry.getKey(), entry.getValue()));
            }
        }
    }

    @Test
    // This program shows how to convert the above program into
    // dataStream.windowAll(...).apply(...).
    public void testDataStreamWindowAll() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataStream<Tuple3<String, Integer, Integer>> dataA = env.fromCollection(input);
        DataStream<Tuple2<String, Integer>> dataB =
                dataA.windowAll(EndOfStreamWindows.get()).apply(new MyAllWindowFunction());
        List<Tuple2<String, Integer>> result = IteratorUtils.toList(dataB.executeAndCollect());

        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }

    private static class MyAllWindowFunction
            implements AllWindowFunction<
                    Tuple3<String, Integer, Integer>, Tuple2<String, Integer>, TimeWindow> {

        @Override
        public void apply(
                TimeWindow timeWindow,
                Iterable<Tuple3<String, Integer, Integer>> iterable,
                Collector<Tuple2<String, Integer>> out)
                throws Exception {
            (new MyGroupReduceFunction()).reduce(iterable, out);
        }
    }
}
