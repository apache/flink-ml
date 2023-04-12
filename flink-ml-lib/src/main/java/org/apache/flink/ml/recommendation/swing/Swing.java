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

package org.apache.flink.ml.recommendation.swing;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * An AlgoOperator which implements the Swing algorithm.
 *
 * <p>Swing is an item recall algorithm. The topology of user-item graph usually can be described as
 * user-item-user or item-user-item, which are like 'swing'. For example, if both user <em>u</em>
 * and user <em>v</em> have purchased the same commodity <em>i</em>, they will form a relationship
 * diagram similar to a swing. If <em>u</em> and <em>v</em> have purchased commodity <em>j</em> in
 * addition to <em>i</em>, it is supposed <em>i</em> and <em>j</em> are similar. The similarity
 * between items in Swing is defined as
 *
 * <p>$$ w_{(i,j)}=\sum_{u\in U_i\cap U_j}\sum_{v\in U_i\cap
 * U_j}{\frac{1}{{(|I_u|+\alpha_1)}^\beta}}*{\frac{1}{{(|I_v|+\alpha_1)}^\beta}}*{\frac{1}{\alpha_2+|I_u\cap
 * I_v|}} $$
 *
 * <p>Note that alpha1 and alpha2 could be zero here. If one of $$|I_u|, |I_v| and |I_u\cap I_v|$$
 * is zero, then the similarity of <em>i</em> and <em>j</em> is zero.
 *
 * <p>See "<a href="https://arxiv.org/pdf/2010.05525.pdf">Large Scale Product Graph Construction for
 * Recommendation in E-commerce</a>" by Xiaoyong Yang, Yadong Zhu and Yi Zhang.
 */
public class Swing implements AlgoOperator<Swing>, SwingParams<Swing> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Swing() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        final String userCol = getUserCol();
        final String itemCol = getItemCol();
        final ResolvedSchema schema = inputs[0].getResolvedSchema();

        if (!(Types.LONG.equals(TableUtils.getTypeInfoByName(schema, userCol))
                && Types.LONG.equals(TableUtils.getTypeInfoByName(schema, itemCol)))) {
            throw new IllegalArgumentException("The types of user and item must be Long.");
        }

        if (getMaxUserBehavior() < getMinUserBehavior()) {
            throw new IllegalArgumentException(
                    String.format(
                            "The maxUserBehavior must be greater than or equal to minUserBehavior. "
                                    + "The current setting: maxUserBehavior=%d, minUserBehavior=%d.",
                            getMaxUserBehavior(), getMinUserBehavior()));
        }

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        SingleOutputStreamOperator<Tuple2<Long, Long>> purchasingBehavior =
                tEnv.toDataStream(inputs[0])
                        .map(
                                row -> {
                                    Long userId = row.getFieldAs(userCol);
                                    Long itemId = row.getFieldAs(itemCol);
                                    if (userId == null || itemId == null) {
                                        throw new RuntimeException(
                                                "Data of user and item column must not be null.");
                                    }
                                    return Tuple2.of(userId, itemId);
                                })
                        .returns(Types.TUPLE(Types.LONG, Types.LONG));

        SingleOutputStreamOperator<Tuple3<Long, Long, long[]>> userBehavior =
                purchasingBehavior
                        .keyBy(tuple -> tuple.f0)
                        .transform(
                                "collectingUserBehavior",
                                Types.TUPLE(
                                        Types.LONG,
                                        Types.LONG,
                                        PrimitiveArrayTypeInfo.LONG_PRIMITIVE_ARRAY_TYPE_INFO),
                                new CollectingUserBehavior(
                                        getMinUserBehavior(), getMaxUserBehavior()));

        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        new TypeInformation[] {Types.LONG, Types.STRING},
                        new String[] {getItemCol(), getOutputCol()});

        DataStream<Row> output =
                userBehavior
                        .keyBy(tuple -> tuple.f1)
                        .transform(
                                "computingSimilarItems",
                                outputTypeInfo,
                                new ComputingSimilarItems(
                                        getK(),
                                        getMaxUserNumPerItem(),
                                        getMaxUserBehavior(),
                                        getAlpha1(),
                                        getAlpha2(),
                                        getBeta(),
                                        getSeed()));

        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Swing load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    /**
     * Collects user behavior data and appends to the input table.
     *
     * <p>During the process, this operator collects users and all items he/she has purchased, and
     * its input table must be bounded.
     */
    private static class CollectingUserBehavior
            extends AbstractStreamOperator<Tuple3<Long, Long, long[]>>
            implements OneInputStreamOperator<Tuple2<Long, Long>, Tuple3<Long, Long, long[]>>,
                    BoundedOneInput {
        private final int minUserItemInteraction;
        private final int maxUserItemInteraction;

        // Maps a user id to a set of items. Because ListState cannot keep values of type `Set`,
        // we use `Map<Long, String>` with null values instead.
        private Map<Long, Map<Long, String>> userAndPurchasedItems = new HashMap<>();

        private ListState<Map<Long, Map<Long, String>>> userAndPurchasedItemsState;

        private CollectingUserBehavior(int minUserItemInteraction, int maxUserItemInteraction) {
            this.minUserItemInteraction = minUserItemInteraction;
            this.maxUserItemInteraction = maxUserItemInteraction;
        }

        @Override
        public void endInput() {

            userAndPurchasedItems.forEach(
                    (user, items) -> {
                        if (items.size() >= minUserItemInteraction
                                && items.size() <= maxUserItemInteraction) {
                            long[] itemsArray = new long[items.size()];
                            int i = 0;
                            for (Long value : items.keySet()) {
                                itemsArray[i++] = value;
                            }
                            items.forEach(
                                    (item, nullValue) ->
                                            output.collect(
                                                    new StreamRecord<>(
                                                            new Tuple3<>(user, item, itemsArray))));
                        }
                    });

            userAndPurchasedItemsState.clear();
        }

        @Override
        public void processElement(StreamRecord<Tuple2<Long, Long>> element) {
            Tuple2<Long, Long> userAndItem = element.getValue();
            long user = userAndItem.f0;
            long item = userAndItem.f1;
            Map<Long, String> items =
                    userAndPurchasedItems.getOrDefault(user, new LinkedHashMap<>());

            if (items.size() <= maxUserItemInteraction) {
                items.put(item, null);
            }

            userAndPurchasedItems.putIfAbsent(user, items);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            userAndPurchasedItemsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "userAndPurchasedItemsState",
                                            Types.MAP(
                                                    Types.LONG,
                                                    Types.MAP(Types.LONG, Types.STRING))));

            OperatorStateUtils.getUniqueElement(
                            userAndPurchasedItemsState, "userAndPurchasedItemsState")
                    .ifPresent(stat -> userAndPurchasedItems = stat);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            userAndPurchasedItemsState.update(Collections.singletonList(userAndPurchasedItems));
        }
    }

    /** Calculates similarity between items and keep top k similar items of each target item. */
    private static class ComputingSimilarItems extends AbstractStreamOperator<Row>
            implements OneInputStreamOperator<Tuple3<Long, Long, long[]>, Row>, BoundedOneInput {

        private final int k;
        private final int maxUserNumPerItem;
        private final int maxUserBehavior;

        private final int alpha1;
        private final int alpha2;
        private final double beta;

        private static final Character commaDelimiter = ',';
        private static final Character semicolonDelimiter = ';';

        private final Random random;

        private Map<Long, long[]> userAndPurchasedItems = new HashMap<>();
        private Map<Long, List<Long>> itemAndPurchasers = new HashMap<>();

        private ListState<Map<Long, long[]>> userAndPurchasedItemsState;
        private ListState<Map<Long, List<Long>>> itemAndPurchasersState;

        private ComputingSimilarItems(
                int k,
                int maxUserNumPerItem,
                int maxUserBehavior,
                int alpha1,
                int alpha2,
                double beta,
                long seed) {
            this.k = k;
            this.maxUserNumPerItem = maxUserNumPerItem;
            this.maxUserBehavior = maxUserBehavior;
            this.alpha1 = alpha1;
            this.alpha2 = alpha2;
            this.beta = beta;
            this.random = new Random(seed);
        }

        @Override
        public void endInput() throws Exception {

            Map<Long, Double> userWeights = new HashMap<>(userAndPurchasedItems.size());
            userAndPurchasedItems.forEach(
                    (k, v) -> {
                        int count = v.length;
                        userWeights.put(k, calculateWeight(count));
                    });

            long[] interaction = new long[maxUserBehavior];
            for (long mainItem : itemAndPurchasers.keySet()) {
                List<Long> userList = itemAndPurchasers.get(mainItem);
                HashMap<Long, Double> id2swing = new HashMap<>();

                for (int i = 1; i < userList.size(); i++) {
                    long u = userList.get(i);
                    int interactionSize;
                    for (int j = i + 1; j < userList.size(); j++) {
                        long v = userList.get(j);
                        interactionSize =
                                calculateCommonItems(
                                        userAndPurchasedItems.get(u),
                                        userAndPurchasedItems.get(v),
                                        interaction);
                        if (interactionSize == 0) {
                            continue;
                        }
                        double similarity =
                                userWeights.get(u)
                                        * userWeights.get(v)
                                        / (alpha2 + interactionSize);
                        for (int k = 0; k < interactionSize; k++) {
                            long simItem = interaction[k];
                            if (simItem == mainItem) {
                                continue;
                            }
                            double itemSimilarity =
                                    id2swing.getOrDefault(simItem, 0.0) + similarity;
                            id2swing.put(simItem, itemSimilarity);
                        }
                    }
                }

                ArrayList<Tuple2<Long, Double>> itemAndScore = new ArrayList<>();
                id2swing.forEach((key, value) -> itemAndScore.add(Tuple2.of(key, value)));

                itemAndScore.sort((o1, o2) -> Double.compare(o2.f1, o1.f1));

                if (itemAndScore.size() == 0) {
                    continue;
                }

                int itemNums = Math.min(k, itemAndScore.size());
                String itemList =
                        itemAndScore.stream()
                                .sequential()
                                .limit(itemNums)
                                .map(tuple2 -> "" + tuple2.f0 + commaDelimiter + tuple2.f1)
                                .collect(Collectors.joining("" + semicolonDelimiter));
                output.collect(new StreamRecord<>(Row.of(mainItem, itemList)));
            }

            userAndPurchasedItemsState.clear();
            itemAndPurchasersState.clear();
        }

        private double calculateWeight(int size) {
            return (1.0 / Math.pow(alpha1 + size, beta));
        }

        private static int calculateCommonItems(long[] u, long[] v, long[] interaction) {
            int pointerU = 0;
            int pointerV = 0;
            int interactionSize = 0;
            while (pointerU < u.length && pointerV < v.length) {
                if (u[pointerU] == v[pointerV]) {
                    interaction[interactionSize++] = u[pointerU];
                    pointerU++;
                    pointerV++;
                } else if (u[pointerU] < v[pointerV]) {
                    pointerU++;
                } else {
                    pointerV++;
                }
            }
            return interactionSize;
        }

        @Override
        public void processElement(StreamRecord<Tuple3<Long, Long, long[]>> streamRecord)
                throws Exception {
            Tuple3<Long, Long, long[]> tuple3 = streamRecord.getValue();
            long user = tuple3.f0;
            long[] userBehavior = tuple3.f2;
            long mainItem = tuple3.f1;

            if (!userAndPurchasedItems.containsKey(user)) {
                Arrays.sort(userBehavior);
                userAndPurchasedItems.put(user, userBehavior);
            }

            itemAndPurchasers.putIfAbsent(mainItem, new ArrayList<>());
            List<Long> purchasers = itemAndPurchasers.get(mainItem);
            // Use the Reservoir Sampling method to randomly select k purchasers from
            // the stream of records where 1<=k<=maxUserNumPerItem.
            // See https://en.wikipedia.org/wiki/Reservoir_sampling for more information on
            // Reservoir Sampling.
            if (purchasers.size() == 0) {
                purchasers.add(0L);
            }
            long total = purchasers.get(0);
            if (purchasers.size() <= maxUserNumPerItem) {
                purchasers.add(user);
            } else {
                int index = random.nextInt((int) total) + 1;
                if (index <= maxUserNumPerItem) {
                    purchasers.set(index, user);
                }
            }
            purchasers.set(0, ++total);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            userAndPurchasedItemsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "userAndPurchasedItemsState",
                                            Types.MAP(
                                                    Types.LONG,
                                                    PrimitiveArrayTypeInfo
                                                            .LONG_PRIMITIVE_ARRAY_TYPE_INFO)));

            OperatorStateUtils.getUniqueElement(
                            userAndPurchasedItemsState, "userAndPurchasedItemsState")
                    .ifPresent(stat -> userAndPurchasedItems = stat);

            itemAndPurchasersState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "itemAndPurchasersState",
                                            Types.MAP(Types.LONG, Types.LIST(Types.LONG))));

            OperatorStateUtils.getUniqueElement(itemAndPurchasersState, "itemAndPurchasersState")
                    .ifPresent(stat -> itemAndPurchasers = stat);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            userAndPurchasedItemsState.update(Collections.singletonList(userAndPurchasedItems));
            itemAndPurchasersState.update(Collections.singletonList(itemAndPurchasers));
        }
    }
}
