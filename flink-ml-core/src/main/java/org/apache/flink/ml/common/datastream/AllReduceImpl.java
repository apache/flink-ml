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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import java.util.HashMap;
import java.util.Map;

/**
 * Applies all-reduce on a data stream where each partition contains only one double array.
 *
 * <p>AllReduce is a communication primitive widely used in MPI. In this implementation, all workers
 * do reduce on a partition of the whole data and they all get the final reduce result. In detail,
 * we split each double array into chunks of fixed size buffer (32KB by default) and let each
 * subtask handle several chunks.
 *
 * <p>There're mainly three stages:
 * <li>All workers send their partial data to other workers for reduce.
 * <li>All workers do reduce on all data it received and then broadcast partial results to others.
 * <li>All workers merge partial results into final result.
 */
class AllReduceImpl {

    @VisibleForTesting static final int CHUNK_SIZE = 1024 * 4;

    /**
     * Applies allReduceSum on the input data stream. The input data stream is supposed to contain
     * one double array in each worker. The result data stream has the same parallelism as the
     * input, where each worker contains one double array that sums all of the double arrays in the
     * input data stream.
     *
     * <p>We throw exception when one of the following two cases happen:
     * <li>There exists one worker that contains more than one double array.
     * <li>The length of double array is not consistent among all workers.
     *
     * @param input The input data stream.
     * @return The result data stream.
     */
    static DataStream<double[]> allReduceSum(DataStream<double[]> input) {
        // chunkId, originalArrayLength, arrayChunk
        DataStream<Tuple3<Integer, Integer, double[]>> allReduceSend =
                input.flatMap(new AllReduceSend())
                        .setParallelism(input.getParallelism())
                        .name("all-reduce-send");

        // taskId, chunkId, originalArrayLength, arrayChunk
        DataStream<Tuple4<Integer, Integer, Integer, double[]>> allReduceSum =
                allReduceSend
                        .partitionCustom(
                                (chunkId, numPartitions) -> chunkId % numPartitions, x -> x.f0)
                        .transform(
                                "all-reduce-sum",
                                new TupleTypeInfo<>(
                                        BasicTypeInfo.INT_TYPE_INFO,
                                        BasicTypeInfo.INT_TYPE_INFO,
                                        BasicTypeInfo.INT_TYPE_INFO,
                                        PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                                new AllReduceSum())
                        .setParallelism(input.getParallelism())
                        .name("all-reduce-sum");

        return allReduceSum
                .partitionCustom((taskIdx, numPartitions) -> taskIdx % numPartitions, x -> x.f0)
                .transform(
                        "all-reduce-recv",
                        PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO,
                        new AllReduceRecv())
                .setParallelism(input.getParallelism())
                .name("all-reduce-recv");
    }

    /**
     * Splits each double array into multiple chunks and sends each chunk to the corresponding
     * worker.
     */
    private static class AllReduceSend
            extends RichFlatMapFunction<double[], Tuple3<Integer, Integer, double[]>> {

        private boolean hasReceivedOneRecord = false;

        private double[] transferBuffer = new double[CHUNK_SIZE];

        @Override
        public void flatMap(
                double[] inputArray, Collector<Tuple3<Integer, Integer, double[]>> out) {
            if (hasReceivedOneRecord) {
                throw new RuntimeException("The input cannot contain more than one double array.");
            }
            hasReceivedOneRecord = true;
            int numTasks = getRuntimeContext().getNumberOfParallelSubtasks();

            for (int taskId = 0; taskId < numTasks; taskId++) {
                int startChunkId = getStartChunkId(taskId, numTasks, inputArray.length);
                int numChunksToHandle = getNumChunksByTaskId(taskId, numTasks, inputArray.length);
                for (int chunkId = startChunkId;
                        chunkId < numChunksToHandle + startChunkId;
                        chunkId++) {
                    System.arraycopy(
                            inputArray,
                            chunkId * CHUNK_SIZE,
                            transferBuffer,
                            0,
                            getLengthOfChunk(chunkId, inputArray.length));
                    out.collect(Tuple3.of(chunkId, inputArray.length, transferBuffer));
                }
            }
        }
    }

    /**
     * Aggregates partitioned array chunks from other workers and broadcast the aggregated array
     * chunk to each worker.
     */
    private static class AllReduceSum
            extends AbstractStreamOperator<Tuple4<Integer, Integer, Integer, double[]>>
            implements OneInputStreamOperator<
                            Tuple3<Integer, Integer, double[]>,
                            Tuple4<Integer, Integer, Integer, double[]>>,
                    BoundedOneInput {

        /**
         * A map that aggregates the received array chunks. The key is chunkId, the value is
         * (originalArrayLength, aggregatedArrayChunk).
         */
        private Map<Integer, Tuple2<Integer, double[]>> aggregatedArrayChunkByChunkId =
                new HashMap<>();

        @Override
        public void endInput() {
            int numTasks = getRuntimeContext().getNumberOfParallelSubtasks();
            for (Map.Entry<Integer, Tuple2<Integer, double[]>> entry :
                    aggregatedArrayChunkByChunkId.entrySet()) {
                for (int taskId = 0; taskId < numTasks; taskId++) {
                    int chunkId = entry.getKey();
                    int originalArrayLength = entry.getValue().f0;
                    double[] aggregatedArrayChunk = entry.getValue().f1;
                    output.collect(
                            new StreamRecord<>(
                                    Tuple4.of(
                                            taskId,
                                            chunkId,
                                            originalArrayLength,
                                            aggregatedArrayChunk)));
                }
            }
        }

        @Override
        public void processElement(StreamRecord<Tuple3<Integer, Integer, double[]>> streamRecord) {
            Tuple3<Integer, Integer, double[]> record = streamRecord.getValue();
            int chunkId = record.f0;
            int originalArrayLength = record.f1;
            double[] arrayChunk = record.f2;
            if (aggregatedArrayChunkByChunkId.containsKey(chunkId)) {
                if (aggregatedArrayChunkByChunkId.get(chunkId).f0 != originalArrayLength) {
                    throw new RuntimeException("The input double array must have same length.");
                }
                double[] curAggregatedArrayChunk = aggregatedArrayChunkByChunkId.get(chunkId).f1;
                for (int i = 0; i < curAggregatedArrayChunk.length; i++) {
                    curAggregatedArrayChunk[i] += arrayChunk[i];
                }
            } else {
                aggregatedArrayChunkByChunkId.put(
                        chunkId, Tuple2.of(originalArrayLength, arrayChunk));
            }
        }
    }

    /** Organizes the received chunks into the result array. */
    private static class AllReduceRecv extends AbstractStreamOperator<double[]>
            implements OneInputStreamOperator<
                            Tuple4<Integer, Integer, Integer, double[]>, double[]>,
                    BoundedOneInput {

        /** Stores the reduced results. */
        double[] resultArray;

        @Override
        public void endInput() {
            if (null != resultArray) {
                output.collect(new StreamRecord<>(resultArray));
            }
        }

        @Override
        public void processElement(
                StreamRecord<Tuple4<Integer, Integer, Integer, double[]>> streamRecord) {
            Tuple4<Integer, Integer, Integer, double[]> ele = streamRecord.getValue();
            int chunkId = ele.f1;
            int originalArrayLength = ele.f2;
            double[] aggregatedArrayChunk = ele.f3;
            if (null == resultArray) {
                resultArray = new double[originalArrayLength];
            }
            System.arraycopy(
                    aggregatedArrayChunk,
                    0,
                    resultArray,
                    chunkId * CHUNK_SIZE,
                    getLengthOfChunk(chunkId, resultArray.length));
        }
    }

    /**
     * Computes how many chunks is an array with length ${len} going to be split into.
     *
     * @param len Length of the array.
     * @return Number of chunks the array is split into.
     */
    private static int getNumChunks(int len) {
        int div = len / CHUNK_SIZE;
        int mod = len % CHUNK_SIZE;
        return mod == 0 ? div : div + 1;
    }

    /**
     * Computes the length of the last chunk of an array with length ${len}.
     *
     * @param len Length of the array.
     * @return Length of the last chunk.
     */
    private static int getLengthOfChunk(int chunkId, int len) {
        if (chunkId == getNumChunks(len) - 1) {
            int mod = len % CHUNK_SIZE;
            return mod == 0 ? CHUNK_SIZE : mod;
        } else {
            return CHUNK_SIZE;
        }
    }

    /**
     * Computes the index of the first chunk that one task needs to handle.
     *
     * @param taskId Index of the current task.
     * @param numTasks Number of parallel tasks.
     * @param len Length of the array to be reduced.
     * @return Start position of this task.
     */
    private static int getStartChunkId(int taskId, int numTasks, int len) {
        int numChunks = getNumChunks(len);
        int div = numChunks / numTasks;
        int mod = numChunks % numTasks;

        if (taskId >= mod) {
            return div * taskId + mod;
        } else {
            return div * taskId + taskId;
        }
    }

    /**
     * Computes the number of chunks that one task needs to handle.
     *
     * @param taskId Index of the current task.
     * @param parallelism Number of parallel tasks.
     * @param len Length of the array to be reduced.
     * @return Number of chunks this task needs to handle.
     */
    private static int getNumChunksByTaskId(int taskId, int parallelism, int len) {
        int numChunks = getNumChunks(len);
        int div = numChunks / parallelism;
        int mod = numChunks % parallelism;

        if (taskId >= mod) {
            return div;
        } else {
            return div + 1;
        }
    }
}
