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

package org.apache.flink.ml.common.fpgrowth;

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.base.BooleanSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.api.java.typeutils.runtime.TupleSerializer;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/** The class for generating association rules from frequent patterns. */
public class AssociationRule {
    private static final Logger LOG = LoggerFactory.getLogger(AssociationRule.class);

    public static DataStream<Tuple4<int[], int[], Integer, double[]>> extractSingleConsequentRules(
            DataStream<Tuple2<Integer, int[]>> patterns,
            DataStream<Tuple2<Integer, Integer>> itemCounts,
            DataStream<Tuple2<Integer, Integer>> transactionCount,
            final double minConfidence,
            final double minLift,
            final int maxPatternLen) {

        /* preprocess and group the patterns into a format suitable for extracting association rules. */
        DataStream<Tuple5<int[], Integer, Integer, Integer, Boolean>> processedPatterns =
                groupPatternsByConseq(patterns);
        itemCounts = itemCounts.union(transactionCount);

        DataStream<Tuple4<int[], int[], Integer, double[]>> rules =
                processedPatterns
                        .connect(itemCounts.broadcast())
                        .transform(
                                "ExtractRulesOperator",
                                Types.TUPLE(
                                        PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO,
                                        PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO,
                                        Types.INT,
                                        PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                                new ExtractRulesOperator(minLift, minConfidence, maxPatternLen));
        return rules;
    }

    private static class ExtractRulesOperator
            extends AbstractStreamOperator<Tuple4<int[], int[], Integer, double[]>>
            implements TwoInputStreamOperator<
                            Tuple5<int[], Integer, Integer, Integer, Boolean>,
                            Tuple2<Integer, Integer>,
                            Tuple4<int[], int[], Integer, double[]>>,
                    BoundedMultiInput {
        final double minLift;
        final double minConfidence;
        final int maxPatternLen;
        private Map<Integer, Integer> itemCounts;
        private TreeMap<int[], Integer> supportMap;
        private ListStateWithCache<Tuple3<int[], Integer, Boolean>> patternsListState;
        private ListStateWithCache<Tuple2<int[], Integer>> supportMapListState;
        double transactionCount = -1;
        private boolean flag1;
        private boolean flag2;

        ExtractRulesOperator(double minLift, double minConfidence, int maxPatternLen) {
            this.minLift = minLift;
            this.minConfidence = minConfidence;
            this.maxPatternLen = maxPatternLen;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            TypeInformation<Tuple2<int[], Integer>> type =
                    Types.TUPLE(PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO, Types.INT);

            supportMapListState =
                    new ListStateWithCache<>(
                            new TupleSerializer<>(
                                    (Class<Tuple2<int[], Integer>>) (Class<?>) Tuple2.class,
                                    new TypeSerializer[] {
                                        new IntPrimitiveArraySerializer(), new IntSerializer()
                                    }),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());

            patternsListState =
                    new ListStateWithCache<>(
                            new TupleSerializer<>(
                                    (Class<Tuple3<int[], Integer, Boolean>>)
                                            (Class<?>) Tuple3.class,
                                    new TypeSerializer[] {
                                        new IntPrimitiveArraySerializer(),
                                        new IntSerializer(),
                                        new BooleanSerializer()
                                    }),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            supportMapListState.snapshotState(context);
            patternsListState.snapshotState(context);
        }

        @Override
        public void endInput(int i) throws Exception {
            if (1 == i) {
                flag2 = true;
            } else {
                flag1 = true;
            }
            int idx = getRuntimeContext().getIndexOfThisSubtask();

            if (flag1 && flag2) {
                for (int epoch = 1; epoch <= maxPatternLen; epoch++) {
                    boolean finished = true;
                    for (Tuple3<int[], Integer, Boolean> pattern : patternsListState.get()) {
                        int patternLen = pattern.f0.length;
                        if (patternLen > epoch) {
                            finished = false;
                        } else if (patternLen == epoch) {
                            processPattern(pattern);
                        }
                    }
                    fillSupportMap();
                    if (finished) {
                        break;
                    }
                }
            }
        }

        @Override
        public void open() throws Exception {
            super.open();
        }

        @Override
        public void close() throws Exception {
            patternsListState.clear();
            super.close();
        }

        @Override
        public void processElement1(
                StreamRecord<Tuple5<int[], Integer, Integer, Integer, Boolean>> streamRecord)
                throws Exception {
            Tuple5<int[], Integer, Integer, Integer, Boolean> pattern = streamRecord.getValue();
            patternsListState.add(Tuple3.of(pattern.f0, pattern.f1, pattern.f4));
        }

        @Override
        public void processElement2(StreamRecord<Tuple2<Integer, Integer>> streamRecord)
                throws Exception {
            Tuple2<Integer, Integer> itemCount = streamRecord.getValue();
            if (itemCounts == null) {
                itemCounts = new HashMap<>();
            }
            if (itemCount.f0 == -1) {
                this.transactionCount = itemCount.f1;
                return;
            }
            itemCounts.put(itemCount.f0, itemCount.f1);
        }

        private void processPattern(Tuple3<int[], Integer, Boolean> pattern) throws Exception {
            boolean rotated = pattern.f2;
            int[] items = pattern.f0;
            if (rotated) {
                int[] ante = Arrays.copyOfRange(items, 1, items.length);
                int conseq = items[0];
                exportRule(ante, conseq, pattern.f1, output);
            } else {
                for (int i = 0; i < items.length - 1; i++) {
                    int[] ante = new int[items.length - 1];
                    System.arraycopy(items, 0, ante, 0, i);
                    System.arraycopy(items, i + 1, ante, i, items.length - i - 1);
                    int conseq = items[i];
                    exportRule(ante, conseq, pattern.f1, output);
                }
                supportMapListState.add(Tuple2.of(items, pattern.f1));
            }
        }

        private void exportRule(
                int[] x,
                int y,
                int suppXY,
                Output<StreamRecord<Tuple4<int[], int[], Integer, double[]>>> collector) {
            Integer suppX = supportMap.get(x);
            Integer suppY = itemCounts.get(y);
            assert suppX != null && suppY != null;
            assert suppX >= suppXY && suppY >= suppXY;
            assert transactionCount > 0;
            double lift = suppXY * transactionCount / (suppX.doubleValue() * suppY.doubleValue());
            double confidence = suppXY / suppX.doubleValue();
            double support = suppXY / transactionCount;
            if (lift >= minLift && confidence >= minConfidence) {
                collector.collect(
                        new StreamRecord<>(
                                Tuple4.of(
                                        x,
                                        new int[] {y},
                                        suppXY,
                                        new double[] {lift, support, confidence})));
            }
        }

        private void fillSupportMap() {
            if (null == supportMap) {
                supportMap = new TreeMap<>(ExtractRulesOperator::comparatorFunction);
            } else {
                supportMap.clear();
            }
            try {
                for (Tuple2<int[], Integer> itemCount : supportMapListState.get()) {
                    supportMap.put(itemCount.f0, itemCount.f1);
                }
            } catch (Exception e) {
            }
            supportMapListState.clear();
        }

        private static int comparatorFunction(int[] o1, int[] o2) {
            if (o1.length != o2.length) {
                return Integer.compare(o1.length, o2.length);
            }
            for (int i = 0; i < o1.length; i++) {
                if (o1[i] != o2[i]) {
                    return Integer.compare(o1[i], o2[i]);
                }
            }
            return 0;
        }
    }

    private static DataStream<Tuple5<int[], Integer, Integer, Integer, Boolean>>
            groupPatternsByConseq(DataStream<Tuple2<Integer, int[]>> patterns) {

        DataStream<Tuple5<int[], Integer, Integer, Integer, Boolean>> processedPatterns =
                patterns.flatMap(
                                new RichFlatMapFunction<
                                        Tuple2<Integer, int[]>,
                                        Tuple5<int[], Integer, Integer, Integer, Boolean>>() {

                                    @Override
                                    public void flatMap(
                                            Tuple2<Integer, int[]> value,
                                            Collector<
                                                            Tuple5<
                                                                    int[],
                                                                    Integer,
                                                                    Integer,
                                                                    Integer,
                                                                    Boolean>>
                                                    out)
                                            throws Exception {

                                        int[] items = value.f1;
                                        int itemsLen = items.length;
                                        Tuple5 nonRotatedpattern =
                                                Tuple5.of(
                                                        value.f1,
                                                        value.f0,
                                                        items[itemsLen - 1],
                                                        itemsLen,
                                                        false);
                                        out.collect(nonRotatedpattern);
                                        if (items.length > 1) {
                                            int tail = items[itemsLen - 1];
                                            for (int i = itemsLen - 1; i >= 1; i--) {
                                                items[i] = items[i - 1];
                                            }
                                            items[0] = tail;
                                            Tuple5 rotatedpattern =
                                                    Tuple5.of(
                                                            items,
                                                            value.f0,
                                                            items[itemsLen - 1],
                                                            itemsLen,
                                                            true);
                                            out.collect(rotatedpattern);
                                        }
                                    }
                                })
                        .name("process_pattern_for_extract")
                        .keyBy(t5 -> t5.f2)
                        .map(t5 -> t5)
                        .returns(
                                Types.TUPLE(
                                        PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO,
                                        Types.INT,
                                        Types.INT,
                                        Types.INT,
                                        Types.BOOLEAN));
        return processedPatterns;
    }
}
