package org.apache.flink.ml.datastream.examples;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.ml.common.MapPartitionFunctionWrapper;
import org.apache.flink.ml.common.SortFunctionBuilder;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * This test shows how to convert dataSet.sortPartition(...).sortPartition(...) into
 * dataStream.transform(...).
 */
public class DataSetSortPartitionTest extends AbstractTestBase {
    List<Tuple2<Integer, Integer>> input;
    List<Tuple2<Integer, Integer>> expectedOutput;

    @Before
    public void before() {
        input = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            input.add(new Tuple2<>(i, 9 - i));
            input.add(new Tuple2<>(i, 10 - i));
        }
        Collections.shuffle(input);

        expectedOutput = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            expectedOutput.add(new Tuple2<>(i, 9 - i));
            expectedOutput.add(new Tuple2<>(i, 10 - i));
        }
    }

    @Test
    // This is an example program that uses dataSet.sortPartition(...).sortPartition(...).
    public void testDataSetSortPartition() throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        DataSet<Tuple2<Integer, Integer>> dataA = env.fromCollection(input);
        DataSet<Tuple2<Integer, Integer>> dataB =
                dataA.sortPartition(0, Order.ASCENDING).sortPartition(1, Order.ASCENDING);
        List<Tuple2<Integer, Integer>> result = dataB.collect();

        Assert.assertEquals(expectedOutput.size(), result.size());
        for (int i = 0; i < result.size(); i++) {
            Assert.assertEquals(expectedOutput.get(i), result.get(i));
        }
    }

    @Test
    // This program shows how to convert the above program into
    // dataStream.transform(...).
    public void testDataStreamTransform() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        MapPartitionFunction<Tuple2<Integer, Integer>, Tuple2<Integer, Integer>> mySortFunction =
                new SortFunctionBuilder<Tuple2<Integer, Integer>>()
                        .sort(0, Order.ASCENDING)
                        .sort(1, Order.ASCENDING)
                        .build();

        DataStream<Tuple2<Integer, Integer>> dataA = env.fromCollection(input);
        DataStream<Tuple2<Integer, Integer>> dataB =
                dataA.transform(
                        "MySortPartition",
                        TupleTypeInfo.getBasicTupleTypeInfo(Integer.class, Integer.class),
                        new MapPartitionFunctionWrapper<>(
                                "MySortPartition",
                                TupleTypeInfo.getBasicTupleTypeInfo(Integer.class, Integer.class),
                                mySortFunction));
        List<Tuple2<Integer, Integer>> result = IteratorUtils.toList(dataB.executeAndCollect());

        Assert.assertEquals(expectedOutput.size(), result.size());
        for (int i = 0; i < result.size(); i++) {
            Assert.assertEquals(expectedOutput.get(i), result.get(i));
        }
    }
}
