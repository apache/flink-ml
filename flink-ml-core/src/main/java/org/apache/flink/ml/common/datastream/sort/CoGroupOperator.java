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

package org.apache.flink.ml.common.datastream.sort;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.CoGroupFunction;
import org.apache.flink.api.common.typeutils.TypeComparator;
import org.apache.flink.api.common.typeutils.TypePairComparator;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.runtime.RuntimePairComparatorFactory;
import org.apache.flink.configuration.AlgorithmOptions;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.memory.DataOutputSerializer;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.runtime.io.disk.iomanager.IOManager;
import org.apache.flink.runtime.memory.MemoryAllocationException;
import org.apache.flink.runtime.memory.MemoryManager;
import org.apache.flink.runtime.operators.sort.ExternalSorter;
import org.apache.flink.runtime.operators.sort.NonReusingSortMergeCoGroupIterator;
import org.apache.flink.runtime.operators.sort.PushSorter;
import org.apache.flink.runtime.operators.sort.ReusingSortMergeCoGroupIterator;
import org.apache.flink.runtime.operators.util.CoGroupTaskIterator;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.TimestampedCollector;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.MutableObjectIterator;
import org.apache.flink.util.TraversableOnceException;

import java.io.Serializable;
import java.util.Iterator;

/**
 * An operator that implements the co-group logic.
 *
 * @param <IN1> The class type of the first input.
 * @param <IN2> The class type of the second input.
 * @param <KEY> The class type of the key.
 * @param <OUT> The class type of the output values.
 */
public class CoGroupOperator<IN1, IN2, KEY extends Serializable, OUT>
        extends AbstractUdfStreamOperator<OUT, CoGroupFunction<IN1, IN2, OUT>>
        implements TwoInputStreamOperator<IN1, IN2, OUT>, BoundedMultiInput {

    private PushSorter<Tuple2<byte[], StreamRecord<IN1>>> sorterA;
    private PushSorter<Tuple2<byte[], StreamRecord<IN2>>> sorterB;
    private TypeComparator<Tuple2<byte[], StreamRecord<IN1>>> comparatorA;
    private TypeComparator<Tuple2<byte[], StreamRecord<IN2>>> comparatorB;
    private KeySelector<IN1, KEY> keySelectorA;
    private KeySelector<IN2, KEY> keySelectorB;
    private TypeSerializer<Tuple2<byte[], StreamRecord<IN1>>> keyAndValueSerializerA;
    private TypeSerializer<Tuple2<byte[], StreamRecord<IN2>>> keyAndValueSerializerB;
    private TypeSerializer<KEY> keySerializer;
    private DataOutputSerializer dataOutputSerializer;
    private long lastWatermarkTimestamp = Long.MIN_VALUE;
    private int remainingInputNum = 2;

    public CoGroupOperator(CoGroupFunction<IN1, IN2, OUT> function) {
        super(function);
    }

    @Override
    public void setup(
            StreamTask<?, ?> containingTask,
            StreamConfig config,
            Output<StreamRecord<OUT>> output) {
        super.setup(containingTask, config, output);
        ClassLoader userCodeClassLoader = containingTask.getUserCodeClassLoader();
        MemoryManager memoryManager = containingTask.getEnvironment().getMemoryManager();
        IOManager ioManager = containingTask.getEnvironment().getIOManager();

        keySelectorA = config.getStatePartitioner(0, userCodeClassLoader);
        keySelectorB = config.getStatePartitioner(1, userCodeClassLoader);
        keySerializer = config.getStateKeySerializer(userCodeClassLoader);
        int keyLength = keySerializer.getLength();

        TypeSerializer<IN1> typeSerializerA = config.getTypeSerializerIn(0, userCodeClassLoader);
        TypeSerializer<IN2> typeSerializerB = config.getTypeSerializerIn(1, userCodeClassLoader);
        keyAndValueSerializerA = new KeyAndValueSerializer<>(typeSerializerA, keyLength);
        keyAndValueSerializerB = new KeyAndValueSerializer<>(typeSerializerB, keyLength);

        if (keyLength > 0) {
            dataOutputSerializer = new DataOutputSerializer(keyLength);
            comparatorA = new FixedLengthByteKeyComparator<>(keyLength);
            comparatorB = new FixedLengthByteKeyComparator<>(keyLength);
        } else {
            dataOutputSerializer = new DataOutputSerializer(64);
            comparatorA = new VariableLengthByteKeyComparator<>();
            comparatorB = new VariableLengthByteKeyComparator<>();
        }

        ExecutionConfig executionConfig = containingTask.getEnvironment().getExecutionConfig();
        double managedMemoryFraction =
                config.getManagedMemoryFractionOperatorUseCaseOfSlot(
                                ManagedMemoryUseCase.OPERATOR,
                                containingTask.getEnvironment().getTaskConfiguration(),
                                userCodeClassLoader)
                        / 2;
        Configuration jobConfiguration = containingTask.getEnvironment().getJobConfiguration();

        try {
            sorterA =
                    ExternalSorter.newBuilder(
                                    memoryManager,
                                    containingTask,
                                    keyAndValueSerializerA,
                                    comparatorA,
                                    executionConfig)
                            .memoryFraction(managedMemoryFraction)
                            .enableSpilling(
                                    ioManager,
                                    jobConfiguration.get(AlgorithmOptions.SORT_SPILLING_THRESHOLD))
                            .maxNumFileHandles(
                                    jobConfiguration.get(AlgorithmOptions.SPILLING_MAX_FAN))
                            .objectReuse(executionConfig.isObjectReuseEnabled())
                            .largeRecords(
                                    jobConfiguration.get(
                                            AlgorithmOptions.USE_LARGE_RECORDS_HANDLER))
                            .build();
            sorterB =
                    ExternalSorter.newBuilder(
                                    memoryManager,
                                    containingTask,
                                    keyAndValueSerializerB,
                                    comparatorB,
                                    executionConfig)
                            .memoryFraction(managedMemoryFraction)
                            .enableSpilling(
                                    ioManager,
                                    jobConfiguration.get(AlgorithmOptions.SORT_SPILLING_THRESHOLD))
                            .maxNumFileHandles(
                                    jobConfiguration.get(AlgorithmOptions.SPILLING_MAX_FAN))
                            .objectReuse(executionConfig.isObjectReuseEnabled())
                            .largeRecords(
                                    jobConfiguration.get(
                                            AlgorithmOptions.USE_LARGE_RECORDS_HANDLER))
                            .build();
        } catch (MemoryAllocationException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void endInput(int inputId) throws Exception {
        if (inputId == 1) {
            sorterA.finishReading();
            remainingInputNum--;
        } else if (inputId == 2) {
            sorterB.finishReading();
            remainingInputNum--;
        } else {
            throw new RuntimeException("Unknown inputId " + inputId);
        }

        if (remainingInputNum > 0) {
            return;
        }

        MutableObjectIterator<Tuple2<byte[], StreamRecord<IN1>>> iteratorA = sorterA.getIterator();
        MutableObjectIterator<Tuple2<byte[], StreamRecord<IN2>>> iteratorB = sorterB.getIterator();
        TypePairComparator<Tuple2<byte[], StreamRecord<IN1>>, Tuple2<byte[], StreamRecord<IN2>>>
                pairComparator =
                        (new RuntimePairComparatorFactory<
                                        Tuple2<byte[], StreamRecord<IN1>>,
                                        Tuple2<byte[], StreamRecord<IN2>>>())
                                .createComparator12(comparatorA, comparatorB);

        CoGroupTaskIterator<Tuple2<byte[], StreamRecord<IN1>>, Tuple2<byte[], StreamRecord<IN2>>>
                coGroupIterator;
        if (getExecutionConfig().isObjectReuseEnabled()) {
            coGroupIterator =
                    new ReusingSortMergeCoGroupIterator<
                            Tuple2<byte[], StreamRecord<IN1>>, Tuple2<byte[], StreamRecord<IN2>>>(
                            iteratorA,
                            iteratorB,
                            keyAndValueSerializerA,
                            comparatorA,
                            keyAndValueSerializerB,
                            comparatorB,
                            pairComparator);
        } else {
            coGroupIterator =
                    new NonReusingSortMergeCoGroupIterator<
                            Tuple2<byte[], StreamRecord<IN1>>, Tuple2<byte[], StreamRecord<IN2>>>(
                            iteratorA,
                            iteratorB,
                            keyAndValueSerializerA,
                            comparatorA,
                            keyAndValueSerializerB,
                            comparatorB,
                            pairComparator);
        }

        coGroupIterator.open();
        TupleUnwrappingIterator<IN1, byte[]> unWrappediteratorA = new TupleUnwrappingIterator<>();
        TupleUnwrappingIterator<IN2, byte[]> unWrappediteratorB = new TupleUnwrappingIterator<>();

        Output<OUT> timestampedCollector = new TimestampedCollector<>(output);
        while (coGroupIterator.next()) {
            unWrappediteratorA.set(coGroupIterator.getValues1().iterator());
            unWrappediteratorB.set(coGroupIterator.getValues2().iterator());
            userFunction.coGroup(unWrappediteratorA, unWrappediteratorB, timestampedCollector);
        }
        coGroupIterator.close();

        Watermark watermark = new Watermark(lastWatermarkTimestamp);
        if (getTimeServiceManager().isPresent()) {
            getTimeServiceManager().get().advanceWatermark(watermark);
        }
        output.emitWatermark(watermark);
    }

    @Override
    public void processWatermark(Watermark watermark) throws Exception {
        if (lastWatermarkTimestamp > watermark.getTimestamp()) {
            throw new RuntimeException("Invalid watermark");
        }
        lastWatermarkTimestamp = watermark.getTimestamp();
    }

    @Override
    public void close() throws Exception {
        super.close();
        sorterA.close();
        sorterB.close();
    }

    @Override
    public void processElement1(StreamRecord<IN1> streamRecord) throws Exception {
        KEY key = keySelectorA.getKey(streamRecord.getValue());
        keySerializer.serialize(key, dataOutputSerializer);
        byte[] serializedKey = dataOutputSerializer.getCopyOfBuffer();
        dataOutputSerializer.clear();
        sorterA.writeRecord(Tuple2.of(serializedKey, streamRecord));
    }

    @Override
    public void processElement2(StreamRecord<IN2> streamRecord) throws Exception {
        KEY key = keySelectorB.getKey(streamRecord.getValue());
        keySerializer.serialize(key, dataOutputSerializer);
        byte[] serializedKey = dataOutputSerializer.getCopyOfBuffer();
        dataOutputSerializer.clear();
        sorterB.writeRecord(Tuple2.of(serializedKey, streamRecord));
    }

    private static class TupleUnwrappingIterator<T, K>
            implements Iterator<T>, Iterable<T>, java.io.Serializable {

        private static final long serialVersionUID = 1L;

        private K lastKey;
        private Iterator<Tuple2<K, StreamRecord<T>>> iterator;
        private boolean iteratorAvailable;

        public void set(Iterator<Tuple2<K, StreamRecord<T>>> iterator) {
            this.iterator = iterator;
            this.iteratorAvailable = true;
        }

        public K getLastKey() {
            return lastKey;
        }

        @Override
        public boolean hasNext() {
            return iterator.hasNext();
        }

        @Override
        public T next() {
            Tuple2<K, StreamRecord<T>> t = iterator.next();
            this.lastKey = t.f0;
            return t.f1.getValue();
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Iterator<T> iterator() {
            if (iteratorAvailable) {
                iteratorAvailable = false;
                return this;
            } else {
                throw new TraversableOnceException();
            }
        }
    }
}
