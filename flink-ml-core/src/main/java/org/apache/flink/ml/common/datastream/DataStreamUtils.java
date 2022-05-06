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

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.dag.Transformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TypeExtractor;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TimestampedCollector;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.TableException;
import org.apache.flink.util.Collector;

import org.apache.commons.collections.IteratorUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/** Provides utility functions for {@link DataStream}. */
@Internal
public class DataStreamUtils {
    /**
     * Applies allReduceSum on the input data stream. The input data stream is supposed to contain
     * up to one double array in each partition. The result data stream has the same parallelism as
     * the input, where each partition contains one double array that sums all of the double arrays
     * in the input data stream.
     *
     * <p>Note that we throw exception when one of the following two cases happen:
     * <li>There exists one partition that contains more than one double array.
     * <li>The length of the double array is not consistent among all partitions.
     *
     * @param input The input data stream.
     * @return The result data stream.
     */
    public static DataStream<double[]> allReduceSum(DataStream<double[]> input) {
        return AllReduceImpl.allReduceSum(input);
    }

    /**
     * Applies a {@link MapPartitionFunction} on a bounded data stream.
     *
     * @param input The input data stream.
     * @param func The user defined mapPartition function.
     * @param <IN> The class type of the input.
     * @param <OUT> The class type of output.
     * @return The result data stream.
     */
    public static <IN, OUT> DataStream<OUT> mapPartition(
            DataStream<IN> input, MapPartitionFunction<IN, OUT> func) {
        TypeInformation<OUT> resultType =
                TypeExtractor.getMapPartitionReturnTypes(func, input.getType(), null, true);
        return input.transform("mapPartition", resultType, new MapPartitionOperator<>(func))
                .setParallelism(input.getParallelism());
    }

    /**
     * Applies a {@link ReduceFunction} on a bounded data stream. The output stream contains at most
     * one stream record and its parallelism is one.
     *
     * @param input The input data stream.
     * @param func The user defined reduce function.
     * @param <T> The class type of the input.
     * @return The result data stream.
     */
    public static <T> DataStream<T> reduce(DataStream<T> input, ReduceFunction<T> func) {
        DataStream<T> partialReducedStream =
                input.transform("reduce", input.getType(), new ReduceOperator<>(func))
                        .setParallelism(input.getParallelism());
        if (partialReducedStream.getParallelism() == 1) {
            return partialReducedStream;
        } else {
            return partialReducedStream
                    .transform("reduce", input.getType(), new ReduceOperator<>(func))
                    .setParallelism(1);
        }
    }

    /**
     * Performs a uniform sampling over the elements in a bounded data stream.
     *
     * <p>This method takes samples without replacement. If the number of elements in the stream is
     * smaller than expected number of samples, all elements will be included in the sample.
     *
     * @param input The input data stream.
     * @param numSamples The number of elements to be sampled.
     * @param randomSeed The seed to randomly pick elements as sample.
     * @return A data stream containing a list of the sampled elements.
     */
    public static <T> DataStream<T> sample(DataStream<T> input, int numSamples, long randomSeed) {
        int inputParallelism = input.getParallelism();

        return input.transform(
                        "samplingOperator",
                        input.getType(),
                        new SamplingOperator<>(numSamples, randomSeed))
                .setParallelism(inputParallelism)
                .transform(
                        "samplingOperator",
                        input.getType(),
                        new SamplingOperator<>(numSamples, randomSeed))
                .setParallelism(1)
                .map(x -> x, input.getType())
                .setParallelism(inputParallelism);
    }

    /**
     * Sets {Transformation#declareManagedMemoryUseCaseAtOperatorScope(ManagedMemoryUseCase, int)}
     * using the given bytes for {@link ManagedMemoryUseCase#OPERATOR}.
     *
     * <p>This method is in reference to Flink's ExecNodeUtil.setManagedMemoryWeight. The provided
     * bytes should be in the same scale as existing usage in Flink, for example,
     * StreamExecWindowAggregate.WINDOW_AGG_MEMORY_RATIO.
     */
    public static <T> void setManagedMemoryWeight(
            Transformation<T> transformation, long memoryBytes) {
        if (memoryBytes > 0) {
            final int weightInMebibyte = Math.max(1, (int) (memoryBytes >> 20));
            final Optional<Integer> previousWeight =
                    transformation.declareManagedMemoryUseCaseAtOperatorScope(
                            ManagedMemoryUseCase.OPERATOR, weightInMebibyte);
            if (previousWeight.isPresent()) {
                throw new TableException(
                        "Managed memory weight has been set, this should not happen.");
            }
        }
    }

    /**
     * A stream operator to apply {@link MapPartitionFunction} on each partition of the input
     * bounded data stream.
     */
    private static class MapPartitionOperator<IN, OUT>
            extends AbstractUdfStreamOperator<OUT, MapPartitionFunction<IN, OUT>>
            implements OneInputStreamOperator<IN, OUT>, BoundedOneInput {

        private ListStateWithCache<IN> valuesState;

        public MapPartitionOperator(MapPartitionFunction<IN, OUT> mapPartitionFunc) {
            super(mapPartitionFunc);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            valuesState =
                    new ListStateWithCache<>(
                            getOperatorConfig().getTypeSerializerIn(0, getClass().getClassLoader()),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            valuesState.snapshotState(context);
        }

        @Override
        public void processElement(StreamRecord<IN> input) throws Exception {
            valuesState.add(input.getValue());
        }

        @Override
        public void endInput() throws Exception {
            userFunction.mapPartition(valuesState.get(), new TimestampedCollector<>(output));
            valuesState.clear();
        }
    }

    /** A stream operator to apply {@link ReduceFunction} on the input bounded data stream. */
    private static class ReduceOperator<T> extends AbstractUdfStreamOperator<T, ReduceFunction<T>>
            implements OneInputStreamOperator<T, T>, BoundedOneInput {
        /** The temp result of the reduce function. */
        private T result;

        private ListState<T> state;

        public ReduceOperator(ReduceFunction<T> userFunction) {
            super(userFunction);
        }

        @Override
        public void endInput() {
            if (result != null) {
                output.collect(new StreamRecord<>(result));
            }
        }

        @Override
        public void processElement(StreamRecord<T> streamRecord) throws Exception {
            if (result == null) {
                result = streamRecord.getValue();
            } else {
                result = userFunction.reduce(streamRecord.getValue(), result);
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            state =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "state",
                                            getOperatorConfig()
                                                    .getTypeSerializerIn(
                                                            0, getClass().getClassLoader())));
            result = OperatorStateUtils.getUniqueElement(state, "state").orElse(null);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            state.clear();
            if (result != null) {
                state.add(result);
            }
        }
    }

    /**
     * Splits the input data into global batches of batchSize. After splitting, each global batch is
     * further split into local batches for downstream operators with each worker has one batch.
     */
    public static <T> DataStream<T[]> generateBatchData(
            DataStream<T> inputData, final int downStreamParallelism, int batchSize) {
        return inputData
                .countWindowAll(batchSize)
                .apply(new GlobalBatchCreator<>())
                .flatMap(new GlobalBatchSplitter<>(downStreamParallelism))
                .partitionCustom((chunkId, numPartitions) -> chunkId, x -> x.f0)
                .map(
                        new MapFunction<Tuple2<Integer, T[]>, T[]>() {
                            @Override
                            public T[] map(Tuple2<Integer, T[]> integerTuple2) throws Exception {
                                return integerTuple2.f1;
                            }
                        });
    }

    /** Splits the input data into global batches. */
    private static class GlobalBatchCreator<T> implements AllWindowFunction<T, T[], GlobalWindow> {
        @Override
        public void apply(GlobalWindow timeWindow, Iterable<T> iterable, Collector<T[]> collector) {
            List<T> points = IteratorUtils.toList(iterable.iterator());
            collector.collect(points.toArray((T[]) new Object[0]));
        }
    }

    /**
     * An operator that splits a global batch into evenly-sized local batches, and distributes them
     * to downstream operator.
     */
    private static class GlobalBatchSplitter<T>
            implements FlatMapFunction<T[], Tuple2<Integer, T[]>> {
        private final int downStreamParallelism;

        public GlobalBatchSplitter(int downStreamParallelism) {
            this.downStreamParallelism = downStreamParallelism;
        }

        @Override
        public void flatMap(T[] values, Collector<Tuple2<Integer, T[]>> collector) {
            int div = values.length / downStreamParallelism;
            int mod = values.length % downStreamParallelism;

            int offset = 0;
            int i = 0;

            int size = div + 1;
            for (; i < mod; i++) {
                collector.collect(Tuple2.of(i, Arrays.copyOfRange(values, offset, offset + size)));
                offset += size;
            }

            size = div;
            for (; i < downStreamParallelism; i++) {
                collector.collect(Tuple2.of(i, Arrays.copyOfRange(values, offset, offset + size)));
                offset += size;
            }
        }
    }

    /*
     * A stream operator that takes a randomly sampled subset of elements in a bounded data stream.
     */
    private static class SamplingOperator<T> extends AbstractStreamOperator<T>
            implements OneInputStreamOperator<T, T>, BoundedOneInput {
        private final int numSamples;

        private final Random random;

        private ListState<T> samplesState;

        private List<T> samples;

        private ListState<Integer> countState;

        private int count;

        SamplingOperator(int numSamples, long randomSeed) {
            this.numSamples = numSamples;
            this.random = new Random(randomSeed);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            ListStateDescriptor<T> samplesDescriptor =
                    new ListStateDescriptor<>(
                            "samplesState",
                            getOperatorConfig()
                                    .getTypeSerializerIn(0, getClass().getClassLoader()));
            samplesState = context.getOperatorStateStore().getListState(samplesDescriptor);
            samples = new ArrayList<>(numSamples);
            samplesState.get().forEach(samples::add);

            ListStateDescriptor<Integer> countDescriptor =
                    new ListStateDescriptor<>("countState", IntSerializer.INSTANCE);
            countState = context.getOperatorStateStore().getListState(countDescriptor);
            Iterator<Integer> countIterator = countState.get().iterator();
            if (countIterator.hasNext()) {
                count = countIterator.next();
            } else {
                count = 0;
            }
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            samplesState.update(samples);
            countState.update(Collections.singletonList(count));
        }

        @Override
        public void processElement(StreamRecord<T> streamRecord) throws Exception {
            T value = streamRecord.getValue();
            count++;

            if (samples.size() < numSamples) {
                samples.add(value);
            } else {
                int index = random.nextInt(count);
                if (index < numSamples) {
                    samples.set(index, value);
                }
            }
        }

        @Override
        public void endInput() throws Exception {
            for (T sample : samples) {
                output.collect(new StreamRecord<>(sample));
            }
        }
    }
}
