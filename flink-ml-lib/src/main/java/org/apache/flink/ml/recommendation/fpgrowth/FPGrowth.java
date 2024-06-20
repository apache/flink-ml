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

package org.apache.flink.ml.recommendation.fpgrowth;

import org.apache.flink.api.common.functions.AbstractRichFunction;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichFilterFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.fpgrowth.AssociationRule;
import org.apache.flink.ml.common.fpgrowth.FPTree;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.StringUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * An implementation of parallel FP-growth algorithm to mine frequent itemset.
 *
 * <p>For detail descriptions, please refer to: <a
 * href="http://dx.doi.org/10.1145/335191.335372">Han et al., Mining frequent patterns without
 * candidate generation</a>. <a href="https://doi.org/10.1145/1454008.1454027">Li et al., PFP:
 * Parallel FP-growth for query recommendation</a>
 */
public class FPGrowth implements AlgoOperator<FPGrowth>, FPGrowthParams<FPGrowth> {
    private static final Logger LOG = LoggerFactory.getLogger(FPGrowth.class);
    private static final String ITEM_INDEX = "ITEM_INDEX";
    private static final String[] FREQ_PATTERN_OUTPUT_COLS = {
        "items", "support_count", "item_count"
    };
    private static final String[] RULES_OUTPUT_COLS = {
        "rule", "item_count", "lift", "support_percent", "confidence_percent", "transaction_count"
    };

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public FPGrowth() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Table data = inputs[0];
        final String itemColName = getItemsCol();
        final double minLift = getMinLift();
        final double minConfidence = getMinConfidence();
        final int maxPatternLen = getMaxPatternLength();
        final String fieldDelimiter = getFieldDelimiter();
        StreamTableEnvironment tenv =
                (StreamTableEnvironment) ((TableImpl) data).getTableEnvironment();

        DataStream<String[]> itemTokens =
                tenv.toDataStream(data)
                        .map(
                                new MapFunction<Row, String[]>() {

                                    @Override
                                    public String[] map(Row value) throws Exception {
                                        Set<String> itemset = new HashSet<>();
                                        String itemsetStr = (String) value.getField(itemColName);
                                        if (!StringUtils.isNullOrWhitespaceOnly(itemsetStr)) {
                                            String[] splited = itemsetStr.split(fieldDelimiter);
                                            itemset.addAll(Arrays.asList(splited));
                                        }
                                        return itemset.toArray(new String[0]);
                                    }
                                })
                        .name("scan_transaction");

        DataStream<String> items =
                itemTokens
                        .flatMap(
                                new FlatMapFunction<String[], String>() {
                                    @Override
                                    public void flatMap(
                                            String[] strings, Collector<String> collector)
                                            throws Exception {
                                        for (String s : strings) {
                                            collector.collect(s);
                                        }
                                    }
                                })
                        .returns(Types.STRING);

        // Count the total num of transactions.
        DataStream<Tuple2<Integer, Integer>> transactionCount = countRecords(itemTokens);
        // Generate a Datastream of minSupport
        final Double minSupport = getMinSupport();
        final int minSupportThreshold = getMinSupportCount();
        DataStream<Double> minSupportStream =
                calculateMinSupport(tenv, transactionCount, minSupport, minSupportThreshold);
        // Count the total number of each item.
        DataStream<Tuple2<String, Integer>> itemCounts = countItems(items);
        // Drop items with support smaller than requirement.
        final String minSuppoerCountDouble = "MIN_SUPPOER_COUNT_DOUBLE";
        itemCounts =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(itemCounts),
                        Collections.singletonMap(minSuppoerCountDouble, minSupportStream),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.filter(
                                    new RichFilterFunction<Tuple2<String, Integer>>() {
                                        Double minSuppport = null;

                                        @Override
                                        public boolean filter(Tuple2<String, Integer> o)
                                                throws Exception {
                                            if (null == minSuppport) {
                                                minSuppport =
                                                        (double)
                                                                getRuntimeContext()
                                                                        .getBroadcastVariable(
                                                                                minSuppoerCountDouble)
                                                                        .get(0);
                                            }
                                            if (o.f1 < minSupportThreshold) {
                                                return false;
                                            } else {
                                                return true;
                                            }
                                        }
                                    });
                        });

        // Assign items with indices, ordered by their support from high to low.
        DataStream<Tuple3<String, Integer, Integer>> itemCountIndex =
                assignItemIndex(tenv, itemCounts);
        // Assign items with partition id.
        DataStream<Tuple2<Integer, Integer>> itemPid = partitionItems(itemCountIndex);

        DataStream<Tuple2<String, Integer>> itemIndex =
                itemCountIndex
                        .map(t3 -> Tuple2.of(t3.f0, t3.f2))
                        .returns(Types.TUPLE(Types.STRING, Types.INT));
        DataStream<Tuple2<Integer, Integer>> itemCount =
                itemCountIndex
                        .map(t3 -> Tuple2.of(t3.f2, t3.f1))
                        .returns(Types.TUPLE(Types.INT, Types.INT));

        DataStream<int[]> transactions = tokensToIndices(itemTokens, itemIndex);
        DataStream<Tuple2<Integer, int[]>> transactionGroups =
                genCondTransactions(transactions, itemPid);

        // Extract all frequent patterns.
        DataStream<Tuple2<Integer, int[]>> indexPatterns =
                mineFreqPattern(transactionGroups, itemPid, maxPatternLen, minSupportStream);
        DataStream<Row> tokenPatterns = patternIndicesToTokens(indexPatterns, itemIndex);

        // Extract consequent rules from frequent patterns.
        DataStream<Tuple4<int[], int[], Integer, double[]>> rules =
                AssociationRule.extractSingleConsequentRules(
                        indexPatterns,
                        itemCount,
                        transactionCount,
                        minConfidence,
                        minLift,
                        maxPatternLen);
        DataStream<Row> rulesToken = ruleIndexToToken(rules, itemIndex);

        Table patternTable = tenv.fromDataStream(tokenPatterns);
        Table rulesTable = tenv.fromDataStream(rulesToken);

        return new Table[] {patternTable, rulesTable};
    }

    /**
     * Generate items partition. To achieve load balance, we assign to each item a score that
     * represents its estimation of number of nodes in the Fp-tree. Then we greedily partition the
     * items to balance the sum of scores in each partition.
     *
     * @param itemCountIndex A DataStream of tuples of item token, count and id.
     * @return A DataStream of tuples of item id and partition id
     */
    private static DataStream<Tuple2<Integer, Integer>> partitionItems(
            DataStream<Tuple3<String, Integer, Integer>> itemCountIndex) {
        DataStream<Tuple2<Integer, Integer>> partition =
                itemCountIndex.transform(
                        "ComputingPartitionCost",
                        Types.TUPLE(Types.INT, Types.INT),
                        new ComputingPartitionCost());
        return partition;
    }

    private static class ComputingPartitionCost
            extends AbstractStreamOperator<Tuple2<Integer, Integer>>
            implements OneInputStreamOperator<
                            Tuple3<String, Integer, Integer>, Tuple2<Integer, Integer>>,
                    BoundedOneInput {
        List<Tuple2<Integer, Integer>> itemCounts;

        private ComputingPartitionCost() {
            itemCounts = new ArrayList<>();
        }

        @Override
        public void endInput() throws Exception {
            int numPartitions = getRuntimeContext().getNumberOfParallelSubtasks();

            PriorityQueue<Tuple2<Integer, Double>> queue =
                    new PriorityQueue<>(numPartitions, Comparator.comparingDouble(o -> o.f1));

            for (int i = 0; i < numPartitions; i++) {
                queue.add(Tuple2.of(i, 0.0));
            }

            List<Double> scaledItemCount = new ArrayList<>(itemCounts.size());
            for (int i = 0; i < itemCounts.size(); i++) {
                Tuple2<Integer, Integer> item = itemCounts.get(i);
                double pos = (double) (item.f0 + 1) / ((double) itemCounts.size());
                double score = pos * item.f1.doubleValue();
                scaledItemCount.add(score);
            }

            List<Integer> order = new ArrayList<>(itemCounts.size());
            for (int i = 0; i < itemCounts.size(); i++) {
                order.add(i);
            }

            order.sort(
                    (o1, o2) -> {
                        double s1 = scaledItemCount.get(o1);
                        double s2 = scaledItemCount.get(o2);
                        return Double.compare(s2, s1);
                    });

            // greedily assign partition number to each item
            for (int i = 0; i < itemCounts.size(); i++) {
                Tuple2<Integer, Integer> item = itemCounts.get(order.get(i));
                double score = scaledItemCount.get(order.get(i));
                Tuple2<Integer, Double> target = queue.poll();
                int targetPartition = target.f0;
                target.f1 += score;
                queue.add(target);
                output.collect(new StreamRecord<>(Tuple2.of(item.f0, targetPartition)));
            }
        }

        @Override
        public void processElement(StreamRecord<Tuple3<String, Integer, Integer>> streamRecord)
                throws Exception {
            Tuple3<String, Integer, Integer> t3 = streamRecord.getValue();
            itemCounts.add(Tuple2.of(t3.f2, t3.f1));
        }
    }

    /**
     * Generate conditional transactions for each partitions.
     *
     * <p>Scan from the longest substring in a transaction, partition the substring into the group
     * where its last element belongs. If the partition already contains a longer substring, skip
     * it.
     *
     * @param transactions A DataStream of transactions.
     * @param targetPartition A DataStream of tuples of item and partition number.
     * @return substring of transactions and partition number
     */
    private static DataStream<Tuple2<Integer, int[]>> genCondTransactions(
            DataStream<int[]> transactions, DataStream<Tuple2<Integer, Integer>> targetPartition) {
        final String itemPartition = "ITEM_PARTITION";
        Map<String, DataStream<?>> broadcastMap = new HashMap<>(1);
        broadcastMap.put(itemPartition, targetPartition);
        return BroadcastUtils.withBroadcastStream(
                Collections.singletonList(transactions),
                broadcastMap,
                inputLists -> {
                    DataStream transactionStream = inputLists.get(0);
                    return transactionStream.flatMap(
                            new RichFlatMapFunction<int[], Tuple2<Integer, int[]>>() {
                                transient Map<Integer, Integer> partitioner;
                                // list of flags used to skip partition that is not empty
                                transient int[] flags;

                                @Override
                                public void flatMap(
                                        int[] transaction, Collector<Tuple2<Integer, int[]>> out)
                                        throws Exception {
                                    if (null == flags) {
                                        int numPartition =
                                                getRuntimeContext().getNumberOfParallelSubtasks();
                                        this.flags = new int[numPartition];
                                    }
                                    if (null == partitioner) {
                                        List<Tuple2<Integer, Integer>> bc =
                                                getRuntimeContext()
                                                        .getBroadcastVariable(itemPartition);
                                        partitioner = new HashMap<>();
                                        for (Tuple2<Integer, Integer> t2 : bc) {
                                            partitioner.put(t2.f0, t2.f1);
                                        }
                                    }
                                    Arrays.fill(flags, 0);
                                    int cnt = transaction.length;
                                    // starts from the longest substring
                                    for (; cnt > 0; cnt--) {
                                        int lastPos = cnt - 1;
                                        int partition = this.partitioner.get(transaction[lastPos]);
                                        if (flags[partition] == 0) {
                                            List<Integer> condTransaction = new ArrayList<>(cnt);
                                            for (int j = 0; j < cnt; j++) {
                                                condTransaction.add(transaction[j]);
                                            }
                                            int[] tr = new int[condTransaction.size()];
                                            for (int j = 0; j < tr.length; j++) {
                                                tr[j] = condTransaction.get(j);
                                            }
                                            out.collect(Tuple2.of(partition, tr));
                                            flags[partition] = 1;
                                        }
                                    }
                                }
                            });
                });
    }

    /**
     * Mine frequent patterns locally in each partition.
     *
     * @param condTransactions The conditional transactions with partition id.
     * @param partitioner A DataStream of tuples of item id and partition id.
     * @param maxPatternLength Maximum pattern length.
     * @return A DataStream of tuples of count and frequent patterns.
     */
    private static DataStream<Tuple2<Integer, int[]>> mineFreqPattern(
            DataStream<Tuple2<Integer, int[]>> condTransactions,
            DataStream<Tuple2<Integer, Integer>> partitioner,
            int maxPatternLength,
            DataStream<Double> minSupport) {
        condTransactions =
                condTransactions.partitionCustom((chunkId, numPartitions) -> chunkId, x -> x.f0);
        partitioner = partitioner.partitionCustom((chunkId, numPartitions) -> chunkId, x -> x.f1);

        List<DataStream<?>> inputList = new ArrayList<>();
        inputList.add(condTransactions);
        inputList.add(partitioner);
        return BroadcastUtils.withBroadcastStream(
                inputList,
                Collections.singletonMap("MIN_COUNT", minSupport),
                inputLists -> {
                    DataStream condStream = inputLists.get(0);
                    DataStream partitionStream = inputLists.get(1);
                    return condStream
                            .connect(partitionStream)
                            .transform(
                                    "mine-freq-pattern",
                                    Types.TUPLE(
                                            Types.INT,
                                            PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO),
                                    new FpTreeConstructor(maxPatternLength));
                });
    }

    private static class MyRichFunction extends AbstractRichFunction {}

    /* The Operator to construct fp-trees of each worker. */
    // private static class FpTreeConstructor extends AbstractStreamOperator<Tuple2<Integer, int[]>>
    private static class FpTreeConstructor
            extends AbstractUdfStreamOperator<Tuple2<Integer, int[]>, RichFunction>
            implements TwoInputStreamOperator<
                            Tuple2<Integer, int[]>,
                            Tuple2<Integer, Integer>,
                            Tuple2<Integer, int[]>>,
                    BoundedMultiInput {

        private final FPTree tree = new FPTree();
        private boolean input1Ends;
        private boolean input2Ends;
        private int minSupportCnt;
        private final int maxPatternLen;
        private Set<Integer> itemList = new HashSet<>();

        FpTreeConstructor(int maxPatternLength) {
            super(new MyRichFunction());
            maxPatternLen = maxPatternLength;
        }

        @Override
        public void open() throws Exception {
            super.open();
            tree.createTree();
        }

        @Override
        public void endInput(int i) throws Exception {
            if (1 == i) {
                LOG.info("Finished adding transactions.");
                input1Ends = true;
            } else {
                LOG.info("Finished adding items.");
                input2Ends = true;
            }

            if (input1Ends && input2Ends) {
                LOG.info("Start to extract fptrees.");
                endInputs();
            }
        }

        public void endInputs() {
            tree.initialize();
            tree.printProfile();
            minSupportCnt =
                    (int)
                            Math.ceil(
                                    ((Double)
                                            userFunction
                                                    .getRuntimeContext()
                                                    .getBroadcastVariable("MIN_COUNT")
                                                    .get(0)));
            int[] suffices = new int[itemList.size()];
            int i = 0;
            for (Integer item : itemList) {
                suffices[i++] = item;
            }
            tree.extractAll(suffices, minSupportCnt, maxPatternLen, output);
            tree.destroyTree();
            LOG.info("itemList size {}.", itemList.size());
            LOG.info("Finished extracting fptrees.");
        }

        @Override
        public void processElement1(StreamRecord<Tuple2<Integer, int[]>> streamRecord)
                throws Exception {
            tree.addTransaction(streamRecord.getValue().f1);
        }

        @Override
        public void processElement2(StreamRecord<Tuple2<Integer, Integer>> streamRecord)
                throws Exception {
            itemList.add(streamRecord.getValue().f0);
        }
    }

    /**
     * Map indices to string in pattern.
     *
     * @param patterns A DataStream of tuples of frequent patterns (represented as int array) and
     *     support count.
     * @param itemIndex A DataStream of tuples of item token and id.
     * @return A DataStream of frequent patterns, support count and length of the pattern.
     */
    private static DataStream<Row> patternIndicesToTokens(
            DataStream<Tuple2<Integer, int[]>> patterns,
            DataStream<Tuple2<String, Integer>> itemIndex) {

        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {Types.STRING, Types.LONG, Types.LONG},
                        FREQ_PATTERN_OUTPUT_COLS);

        Map<String, DataStream<?>> broadcastMap = new HashMap<>(1);
        broadcastMap.put(ITEM_INDEX, itemIndex);
        return BroadcastUtils.withBroadcastStream(
                Collections.singletonList(patterns),
                broadcastMap,
                inputList -> {
                    DataStream freqPatterns = inputList.get(0);
                    return freqPatterns
                            .map(
                                    new RichMapFunction<Tuple2<Integer, int[]>, Row>() {
                                        Map<Integer, String> tokenToId;

                                        @Override
                                        public Row map(Tuple2<Integer, int[]> pattern)
                                                throws Exception {
                                            if (null == tokenToId) {
                                                tokenToId = new HashMap<>();
                                                List<Tuple2<String, Integer>> itemIndexList =
                                                        getRuntimeContext()
                                                                .getBroadcastVariable("ITEM_INDEX");
                                                for (Tuple2<String, Integer> t2 : itemIndexList) {
                                                    tokenToId.put(t2.f1, t2.f0);
                                                }
                                            }
                                            int len = pattern.f1.length;
                                            if (len == 0) {
                                                return null;
                                            }
                                            StringBuilder sbd = new StringBuilder();
                                            sbd.append(tokenToId.get(pattern.f1[0]));
                                            for (int i = 1; i < len; i++) {
                                                sbd.append(",")
                                                        .append(tokenToId.get(pattern.f1[i]));
                                            }
                                            return Row.of(
                                                    sbd.toString(), (long) pattern.f0, (long) len);
                                        }
                                    })
                            .name("flatMap_id_to_token")
                            .returns(outputTypeInfo);
                });
    }

    /**
     * Count the total rows of input Datastream.
     *
     * @param itemTokens A DataStream of input transactions.
     * @return A DataStream of one record, recording the number of input rows.
     */
    private static DataStream<Tuple2<Integer, Integer>> countRecords(
            DataStream<String[]> itemTokens) {
        return DataStreamUtils.aggregate(
                        itemTokens,
                        new AggregateFunction<String[], Integer, Integer>() {

                            @Override
                            public Integer createAccumulator() {
                                return 0;
                            }

                            @Override
                            public Integer add(String[] strings, Integer count) {
                                if (strings.length > 0) {
                                    return count + 1;
                                }
                                return count;
                            }

                            @Override
                            public Integer getResult(Integer count) {
                                return count;
                            }

                            @Override
                            public Integer merge(Integer count, Integer acc1) {
                                return count + acc1;
                            }
                        })
                .map(
                        new MapFunction<Integer, Tuple2<Integer, Integer>>() {
                            @Override
                            public Tuple2<Integer, Integer> map(Integer count) throws Exception {
                                return Tuple2.of(-1, count);
                            }
                        })
                .name("count_transaction")
                .returns(Types.TUPLE(Types.INT, Types.INT));
    }

    /**
     * Count the number of occurence of each item.
     *
     * @param items A DataStream of items.
     * @return A DataStream of tuples of item string and count.
     */
    private static DataStream<Tuple2<String, Integer>> countItems(DataStream<String> items) {
        return DataStreamUtils.keyedAggregate(
                items.keyBy(s -> s),
                new AggregateFunction<String, Tuple2<String, Integer>, Tuple2<String, Integer>>() {

                    @Override
                    public Tuple2<String, Integer> createAccumulator() {
                        return Tuple2.of(null, 0);
                    }

                    @Override
                    public Tuple2<String, Integer> add(String item, Tuple2<String, Integer> acc) {
                        if (null == acc.f0) {
                            acc.f0 = item;
                        }
                        acc.f1++;
                        return acc;
                    }

                    @Override
                    public Tuple2<String, Integer> getResult(Tuple2<String, Integer> t2) {
                        return t2;
                    }

                    @Override
                    public Tuple2<String, Integer> merge(
                            Tuple2<String, Integer> acc1, Tuple2<String, Integer> acc2) {
                        acc2.f1 += acc1.f1;
                        return acc2;
                    }
                },
                Types.TUPLE(Types.STRING, Types.INT),
                Types.TUPLE(Types.STRING, Types.INT));
    }

    /**
     * Calculate minimal support count of the frequent pattern. If minSupportRate is not null,
     * return minSupportRate * transactionCount, else return minSupportCount
     *
     * @param tenv
     * @param transactionCount
     * @param minSupportRate
     * @param minSupportCount
     * @return A DataStream of one record, recording the minimal support.
     */
    private static DataStream<Double> calculateMinSupport(
            StreamTableEnvironment tenv,
            DataStream<Tuple2<Integer, Integer>> transactionCount,
            final Double minSupportRate,
            final int minSupportCount) {
        final String supportCount = "MIN_SUPPORT_COUNT";
        final String supportRate = "MIN_SUPPORT_RATE";
        Map<String, DataStream<?>> bc = new HashMap<>(2);
        DataStream<Row> minSupportCountStream = tenv.toDataStream(tenv.fromValues(minSupportCount));
        bc.put(supportCount, minSupportCountStream);
        DataStream<Row> minSupportRateStream = tenv.toDataStream(tenv.fromValues(minSupportRate));
        bc.put(supportRate, minSupportRateStream);
        return BroadcastUtils.withBroadcastStream(
                Collections.singletonList(transactionCount),
                bc,
                inputLists -> {
                    DataStream transactionCountStream = inputLists.get(0);
                    return transactionCountStream.map(
                            new RichMapFunction<Tuple2<Integer, Integer>, Double>() {
                                @Override
                                public Double map(Tuple2<Integer, Integer> tuple2)
                                        throws Exception {
                                    Double bcSupport =
                                            ((Row)
                                                            getRuntimeContext()
                                                                    .getBroadcastVariable(
                                                                            supportRate)
                                                                    .get(0))
                                                    .getFieldAs(0);
                                    int bcCount =
                                            ((Row)
                                                            getRuntimeContext()
                                                                    .getBroadcastVariable(
                                                                            supportCount)
                                                                    .get(0))
                                                    .getFieldAs(0);
                                    if (bcCount > 0) {
                                        return (double) bcCount;
                                    }
                                    return tuple2.f1 * bcSupport;
                                }
                            });
                });
    }

    /**
     * Map item token to indice based on its descending order of count.
     *
     * @param tenv the StreamTableEnvironment of execution environment.
     * @param itemCounts A DataStream of tuples of item and count.
     * @return A DataStream of tuples of item and count and index.
     */
    private static DataStream<Tuple3<String, Integer, Integer>> assignItemIndex(
            StreamTableEnvironment tenv, DataStream<Tuple2<String, Integer>> itemCounts) {
        final String itemSupports = "ITEM_SUPPORTS";

        return BroadcastUtils.withBroadcastStream(
                Collections.singletonList(
                        tenv.toDataStream(tenv.fromValues(Collections.singletonMap(-1, -1)))),
                Collections.singletonMap(itemSupports, itemCounts),
                inputList -> {
                    DataStream input = inputList.get(0);
                    return input.flatMap(
                                    new RichFlatMapFunction<
                                            Row, Tuple3<String, Integer, Integer>>() {
                                        List<Tuple2<String, Integer>> supportCount;

                                        @Override
                                        public void flatMap(
                                                Row o,
                                                Collector<Tuple3<String, Integer, Integer>>
                                                        collector)
                                                throws Exception {
                                            if (null == supportCount) {
                                                supportCount =
                                                        getRuntimeContext()
                                                                .getBroadcastVariable(itemSupports);
                                            }
                                            Integer[] order = new Integer[supportCount.size()];
                                            for (int i = 0; i < order.length; i++) {
                                                order[i] = i;
                                            }
                                            Arrays.sort(
                                                    order,
                                                    new Comparator<Integer>() {
                                                        @Override
                                                        public int compare(Integer o1, Integer o2) {
                                                            Integer cnt1 = supportCount.get(o1).f1;
                                                            Integer cnt2 = supportCount.get(o2).f1;
                                                            if (cnt1.equals(cnt2)) {
                                                                return supportCount
                                                                        .get(o1)
                                                                        .f0
                                                                        .compareTo(
                                                                                supportCount.get(o2)
                                                                                        .f0);
                                                            }
                                                            return Integer.compare(cnt2, cnt1);
                                                        }
                                                    });
                                            for (int i = 0; i < order.length; i++) {
                                                collector.collect(
                                                        Tuple3.of(
                                                                supportCount.get(order[i]).f0,
                                                                supportCount.get(order[i]).f1,
                                                                i));
                                            }
                                        }
                                    })
                            .name("item_indexer");
                });
    }

    /**
     * Map string to indices in transactions.
     *
     * @param itemSets A DataStream of transactions.
     * @param itemIndex A DataStream of tuples of item token and id.
     * @return A DataStream of tuples of transactions represented as int array
     */
    private static DataStream<int[]> tokensToIndices(
            DataStream<String[]> itemSets, DataStream<Tuple2<String, Integer>> itemIndex) {
        Map<String, DataStream<?>> broadcastMap = new HashMap<>(1);
        broadcastMap.put(ITEM_INDEX, itemIndex);
        return BroadcastUtils.withBroadcastStream(
                Collections.singletonList(itemSets),
                broadcastMap,
                inputList -> {
                    DataStream transactions = inputList.get(0);
                    return transactions
                            .map(
                                    new RichMapFunction<String[], int[]>() {
                                        Map<String, Integer> tokenToId;

                                        @Override
                                        public int[] map(String[] transaction) throws Exception {
                                            if (null == tokenToId) {
                                                LOG.info("Trying to get ITEM_INDEX.");
                                                tokenToId = new HashMap<>();
                                                List<Tuple2<String, Integer>> itemIndexList =
                                                        getRuntimeContext()
                                                                .getBroadcastVariable("ITEM_INDEX");
                                                for (Tuple2<String, Integer> t2 : itemIndexList) {
                                                    tokenToId.put(t2.f0, t2.f1);
                                                }
                                                LOG.info(
                                                        "Size of tokenToId is {}.",
                                                        tokenToId.size());
                                            }
                                            int[] items = new int[transaction.length];
                                            int len = 0;
                                            for (String item : transaction) {
                                                Integer id = tokenToId.get(item);
                                                if (id != null) {
                                                    items[len++] = id;
                                                }
                                            }
                                            if (len > 0) {
                                                int[] qualified = Arrays.copyOfRange(items, 0, len);
                                                Arrays.sort(qualified);
                                                return qualified;
                                            } else {
                                                return new int[0];
                                            }
                                        }
                                    })
                            .name("map_token_to_index")
                            .returns(PrimitiveArrayTypeInfo.INT_PRIMITIVE_ARRAY_TYPE_INFO);
                });
    }

    /**
     * Map indices to string in association rules.
     *
     * @param rules A DataStream of tuples of antecedent, consequent, support count, [lift, support,
     *     confidence].
     * @param itemIndex A DataStream of tuples of item token and id.
     * @return A DataStream of tuples of row of rules, length of the rule, lift, support, confidence
     *     and support count.
     */
    private static DataStream<Row> ruleIndexToToken(
            DataStream<Tuple4<int[], int[], Integer, double[]>> rules,
            DataStream<Tuple2<String, Integer>> itemIndex) {

        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {
                            Types.STRING,
                            Types.LONG,
                            Types.DOUBLE,
                            Types.DOUBLE,
                            Types.DOUBLE,
                            Types.LONG
                        },
                        RULES_OUTPUT_COLS);

        Map<String, DataStream<?>> broadcastMap = new HashMap<>(1);
        broadcastMap.put(ITEM_INDEX, itemIndex);
        return BroadcastUtils.withBroadcastStream(
                Collections.singletonList(rules),
                broadcastMap,
                inputList -> {
                    DataStream freqPatterns = inputList.get(0);
                    return freqPatterns
                            .map(
                                    new RichMapFunction<
                                            Tuple4<int[], int[], Integer, double[]>, Row>() {
                                        Map<Integer, String> tokenToId;

                                        @Override
                                        public Row map(Tuple4<int[], int[], Integer, double[]> rule)
                                                throws Exception {
                                            if (null == tokenToId) {
                                                tokenToId = new HashMap<>();
                                                List<Tuple2<String, Integer>> itemIndexList =
                                                        getRuntimeContext()
                                                                .getBroadcastVariable("ITEM_INDEX");
                                                for (Tuple2<String, Integer> t2 : itemIndexList) {
                                                    tokenToId.put(t2.f1, t2.f0);
                                                }
                                            }
                                            StringBuilder sbd = new StringBuilder();
                                            int[] ascent = rule.f0;
                                            int[] consq = rule.f1;
                                            sbd.append(tokenToId.get(ascent[0]));
                                            for (int i = 1; i < ascent.length; i++) {
                                                sbd.append(",").append(tokenToId.get(ascent[i]));
                                            }
                                            sbd.append("=>");
                                            sbd.append(tokenToId.get(consq[0]));
                                            for (int i = 1; i < consq.length; i++) {
                                                sbd.append(",").append(tokenToId.get(consq[i]));
                                            }
                                            return Row.of(
                                                    sbd.toString(),
                                                    (long) (ascent.length + consq.length),
                                                    rule.f3[0],
                                                    rule.f3[1],
                                                    rule.f3[2],
                                                    (long) rule.f2);
                                        }
                                    })
                            .returns(outputTypeInfo);
                });
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static FPGrowth load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
