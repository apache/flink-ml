package org.apache.flink.ml.datastream.examples;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.ml.common.MapPartitionFunctionWrapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.util.Collector;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/** This test shows how to convert dataSet.mapPartition(...) into dataStream.transform(...). */
public class DataSetMapPartitionTest extends AbstractTestBase {
    List<Tuple2<String, String>> input;
    List<Tuple2<String, Integer>> expectedOutput;

    @Before
    public void before() {
        String inputStr =
                "1 1\n2 2\n2 8\n4 4\n4 4\n6 6\n7 7\n8 8\n"
                        + "1 1\n2 2\n2 2\n4 4\n4 4\n6 3\n5 9\n8 8\n1 1\n2 2\n2 2\n3 0\n4 4\n"
                        + "5 9\n7 7\n8 8\n1 1\n9 1\n5 9\n4 4\n4 4\n6 6\n7 7\n8 8\n";
        String expectedOutputStr =
                "1 11\n2 12\n4 14\n4 14\n1 11\n2 12\n2 12\n4 14\n4 14\n3 16\n1 11\n2 12\n2 12\n0 13\n4 14\n1 11\n4 14\n4 14\n";

        input = new ArrayList<>();
        for (String s : inputStr.split("\n")) {
            String[] fields = s.split(" ");
            input.add(new Tuple2<>(fields[0], fields[1]));
        }

        expectedOutput = new ArrayList<>();
        for (String s : expectedOutputStr.split("\n")) {
            String[] fields = s.split(" ");
            expectedOutput.add(new Tuple2<>(fields[0], Integer.parseInt(fields[1])));
        }
    }

    @Test
    // This is an example program that uses dataSet.mapPartition(...)
    public void testDataSetMapPartition() throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataSet<Tuple2<String, String>> dataA = env.fromCollection(input);
        DataSet<Tuple2<String, Integer>> dataB = dataA.mapPartition(new MyMapPartitionFunction());

        List<Tuple2<String, Integer>> result = dataB.collect();
        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }

    private static class MyMapPartitionFunction
            implements MapPartitionFunction<Tuple2<String, String>, Tuple2<String, Integer>> {

        @Override
        public void mapPartition(
                Iterable<Tuple2<String, String>> values, Collector<Tuple2<String, Integer>> out) {
            for (Tuple2<String, String> value : values) {
                String keyString = value.f0;
                String valueString = value.f1;

                int keyInt = Integer.parseInt(keyString);
                int valueInt = Integer.parseInt(valueString);

                if (keyInt + valueInt < 10) {
                    out.collect(new Tuple2<>(valueString, keyInt + 10));
                }
            }
        }
    }

    @Test
    // This program shows how to convert dataSet.mapPartition(...) into dataStream.transform(...).
    // This could be used if the processing logic needs to be applied on multiple elements at a
    // time, or if the processing logic needs to be applied exactly once on every subtask.
    public void testDataStreamTransform() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);

        DataStream<Tuple2<String, String>> dataA = env.fromCollection(input);
        DataStream<Tuple2<String, Integer>> dataB =
                dataA.transform(
                        "StreamOperatorExample",
                        TupleTypeInfo.getBasicTupleTypeInfo(String.class, Integer.class),
                        new MapPartitionFunctionWrapper<>(
                                "StreamOperatorExample",
                                TupleTypeInfo.getBasicTupleTypeInfo(String.class, String.class),
                                new MyMapPartitionFunction()));
        List<Tuple2<String, Integer>> result = IteratorUtils.toList(dataB.executeAndCollect());

        compareResultCollections(expectedOutput, result, new TestBaseUtils.TupleComparator<>());
    }
}
