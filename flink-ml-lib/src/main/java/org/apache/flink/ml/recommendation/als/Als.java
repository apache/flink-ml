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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.api.java.typeutils.GenericTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.common.ps.iterations.AllReduceStage;
import org.apache.flink.ml.common.ps.iterations.IterationStageList;
import org.apache.flink.ml.common.ps.iterations.PullStage;
import org.apache.flink.ml.common.ps.iterations.PushStage;
import org.apache.flink.ml.common.ps.utils.TrainingUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableFunction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the Als algorithm.
 *
 * <p>ALS tries to decompose a matrix R as R = X * Yt. Here X and Y are called factor matrices.
 * Matrix R is usually a sparse matrix representing ratings given from users to items. ALS tries to
 * find X and Y that minimize || R - X * Yt ||^2. This is done by iterations. At each step, X is
 * fixed and Y is solved, then Y is fixed and X is solved.
 *
 * <p>The algorithm is described in "Large-scale Parallel Collaborative Filtering for the Netflix
 * Prize, 2007". This algorithm also supports implicit preference model described in "Collaborative
 * Filtering for Implicit Feedback Datasets, 2008".
 */
public class Als implements Estimator<Als, AlsModel>, AlsParams<Als> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final int THRESHOLD = 100000;

    public Als() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public AlsModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        final String userCol = getUserCol();
        final String itemCol = getItemCol();
        final String ratingCol = getRatingCol();

        DataStream<Row> trainData = tEnv.toDataStream(inputs[0]);

        DataStream<Tuple3<Long, Long, Double>> alsInput =
                trainData
                        .map(
                                (MapFunction<Row, Tuple3<Long, Long, Double>>)
                                        value -> {
                                            Long user = value.getFieldAs(userCol);
                                            Long item = value.getFieldAs(itemCol);
                                            user = 2L * user;
                                            item = 2L * item + 1L;
                                            Number rating =
                                                    ratingCol == null
                                                            ? 0.0F
                                                            : value.getFieldAs(ratingCol);

                                            return new Tuple3<>(user, item, rating.doubleValue());
                                        })
                        .name("generateInputALsData")
                        .returns(Types.TUPLE(Types.LONG, Types.LONG, Types.DOUBLE));

        /* Initializes variables before iteration. */
        DataStream<Ratings> ratingData = initRatings(alsInput);
        int parallelism = ratingData.getParallelism();
        AlsMLSession mlSession = new AlsMLSession(getImplicitPrefs(), getRank(), parallelism);
        ExecutionConfig executionConfig = ratingData.getExecutionConfig();
        TypeSerializer<double[]> typeSerializer =
                TypeInformation.of(double[].class).createSerializer(executionConfig);

        IterationStageList<AlsMLSession> iterationStages =
                constructIterationStage(mlSession, typeSerializer);

        AlsModelUpdater updater = new AlsModelUpdater(getRank());
        DataStreamList resultList =
                TrainingUtils.train(
                        ratingData,
                        iterationStages,
                        new TupleTypeInfo<>(
                                Types.LONG,
                                PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO),
                        updater,
                        Math.max(1, parallelism / 2));

        DataStream<Tuple2<Long, double[]>> returnData = resultList.get(0);

        DataStream<AlsModelData> modelData =
                returnData
                        .transform(
                                "generateModelData",
                                TypeInformation.of(AlsModelData.class),
                                new GenerateModelData())
                        .name("generateModelData");

        AlsModel model = new AlsModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    private IterationStageList<AlsMLSession> constructIterationStage(
            AlsMLSession mlSession, TypeSerializer<double[]> typeSerializer) {
        IterationStageList<AlsMLSession> iterationStages = new IterationStageList<>(mlSession);
        if (getImplicitPrefs()) {
            /*
             * If using implicit prefs, the whole yty matrix must be computed by all reduce stage.
             */
            iterationStages
                    .addStage(new ComputeYtyIndices())
                    .addStage(
                            new PullStage(
                                    () -> mlSession.pullIndices,
                                    () -> mlSession.aggregatorSDAArray,
                                    new YtyAggregator()))
                    .addStage(new CopyAllReduceData(getRank()))
                    .addStage(
                            new AllReduceStage<>(
                                    () -> mlSession.allReduceBuffer,
                                    () -> mlSession.allReduceBuffer,
                                    (ReduceFunction<double[][]>) Als::sumYty,
                                    typeSerializer,
                                    1));
        }

        iterationStages
                .addStage(new ComputeNeighborIndices(getRank()))
                .addStage(new PullStage(() -> mlSession.pullIndices, () -> mlSession.pullValues))
                .addStage(
                        new UpdateCommonFactors(
                                getRank(),
                                getImplicitPrefs(),
                                getNonNegative(),
                                getRegParam(),
                                getAlpha()))
                .addStage(new PushStage(() -> mlSession.pushIndices, () -> mlSession.pushValues));

        iterationStages
                .addStage(
                        new ComputeLsMatrixVector(
                                getRank(), getImplicitPrefs(), getRegParam(), getAlpha()))
                .addStage(new PushStage(() -> mlSession.pushIndices, () -> mlSession.pushValues))
                .addStage(new PullStage(() -> mlSession.pullIndices, () -> mlSession.pullValues))
                .addStage(new UpdateHotPointFactors(getRank(), getNonNegative()))
                .addStage(new PushStage(() -> mlSession.pushIndices, () -> mlSession.pushValues));

        iterationStages.setTerminationCriteria(
                (SerializableFunction<AlsMLSession, Boolean>)
                        o -> o.iterationId / (o.numItemBlocks + o.numUserBlocks) >= getMaxIter());
        return iterationStages;
    }

    /** Generates the ModelData from the results of iteration. */
    private static class GenerateModelData extends AbstractStreamOperator<AlsModelData>
            implements OneInputStreamOperator<Tuple2<Long, double[]>, AlsModelData>,
                    BoundedOneInput {

        private final List<Tuple2<Long, float[]>> userFactors = new ArrayList<>();
        private final List<Tuple2<Long, float[]>> itemFactors = new ArrayList<>();

        @Override
        public void endInput() throws Exception {
            LOG.info("Generates model   ... " + System.currentTimeMillis());
            output.collect(new StreamRecord<>(new AlsModelData(userFactors, itemFactors)));
        }

        @Override
        public void processElement(StreamRecord<Tuple2<Long, double[]>> streamRecord)
                throws Exception {
            Tuple2<Long, double[]> t2 = streamRecord.getValue();

            if (t2.f0 % 2L == 1L) {
                long id = (t2.f0 - 1) / 2L;
                float[] factor = new float[t2.f1.length];
                for (int i = 0; i < factor.length; ++i) {
                    factor[i] = (float) t2.f1[i];
                }
                itemFactors.add(Tuple2.of(id, factor));
            } else {
                long id = t2.f0 / 2L;
                float[] factor = new float[t2.f1.length];
                for (int i = 0; i < factor.length; ++i) {
                    factor[i] = (float) t2.f1[i];
                }
                userFactors.add(Tuple2.of(id, factor));
            }
        }
    }

    /**
     * Initializes the ratings data with the input graph.
     *
     * @param alsInput The input graph.
     * @return The ratings data.
     */
    private DataStream<Ratings> initRatings(DataStream<Tuple3<Long, Long, Double>> alsInput) {

        DataStream<Ratings> ratings =
                alsInput.flatMap(
                                new RichFlatMapFunction<
                                        Tuple3<Long, Long, Double>, Tuple3<Long, Long, Double>>() {

                                    @Override
                                    public void flatMap(
                                            Tuple3<Long, Long, Double> value,
                                            Collector<Tuple3<Long, Long, Double>> out) {
                                        out.collect(Tuple3.of(value.f0, value.f1, value.f2));
                                        out.collect(Tuple3.of(value.f1, value.f0, value.f2));
                                    }
                                })
                        .keyBy((KeySelector<Tuple3<Long, Long, Double>, Long>) value -> value.f0)
                        .window(EndOfStreamWindows.get())
                        .process(
                                new ProcessWindowFunction<
                                        Tuple3<Long, Long, Double>, Ratings, Long, TimeWindow>() {

                                    @Override
                                    public void process(
                                            Long o,
                                            Context context,
                                            Iterable<Tuple3<Long, Long, Double>> iterable,
                                            Collector<Ratings> collector) {
                                        long srcNodeId = -1L;
                                        List<Tuple2<Long, Double>> neighbors = new ArrayList<>();

                                        for (Tuple3<Long, Long, Double> t4 : iterable) {
                                            srcNodeId = t4.f0;
                                            neighbors.add(Tuple2.of(t4.f1, t4.f2));
                                        }
                                        if (neighbors.size() > THRESHOLD) {
                                            int numBlock =
                                                    neighbors.size() / THRESHOLD
                                                            + (neighbors.size() % THRESHOLD == 0L
                                                                    ? 0
                                                                    : 1);
                                            int blockSize = neighbors.size() / numBlock;
                                            int startIndex = 0;
                                            for (int i = 0; i < numBlock; ++i) {
                                                Ratings tmpRating = new Ratings();
                                                int offset =
                                                        Math.min(
                                                                i + 1, neighbors.size() % numBlock);
                                                int endIndex =
                                                        Math.min(
                                                                neighbors.size(),
                                                                (i + 1) * blockSize + offset);
                                                int size = endIndex - startIndex;
                                                tmpRating.neighbors = new long[size];
                                                tmpRating.scores = new double[size];
                                                for (int j = 0; j < size; j++) {
                                                    tmpRating.neighbors[j] =
                                                            neighbors.get(startIndex + j).f0;
                                                    tmpRating.scores[j] =
                                                            neighbors.get(startIndex + j).f1;
                                                }
                                                startIndex = endIndex;
                                                tmpRating.nodeId = srcNodeId;
                                                tmpRating.isMainNode = (i == 0);
                                                tmpRating.isSplit = true;
                                                tmpRating.numNeighbors = neighbors.size();
                                                collector.collect(tmpRating);
                                            }
                                        } else {
                                            Ratings returnRatings = new Ratings();
                                            returnRatings.nodeId = srcNodeId;
                                            returnRatings.neighbors = new long[neighbors.size()];
                                            returnRatings.scores = new double[neighbors.size()];
                                            returnRatings.isSplit = false;
                                            returnRatings.numNeighbors = neighbors.size();
                                            returnRatings.isMainNode = false;
                                            for (int i = 0;
                                                    i < returnRatings.neighbors.length;
                                                    i++) {
                                                returnRatings.neighbors[i] = neighbors.get(i).f0;
                                                returnRatings.scores[i] = neighbors.get(i).f1;
                                            }
                                            collector.collect(returnRatings);
                                        }
                                    }
                                })
                        .returns(GenericTypeInfo.of(Ratings.class))
                        .name("initRatings")
                        .rebalance();
        DataStream<Ratings> profile = generateDataProfile(ratings).broadcast();
        return ratings.union(profile);
    }

    private DataStream<Ratings> generateDataProfile(DataStream<Ratings> ratingData) {
        DataStream<Tuple5<Long, Long, Long, Integer, Integer>> localSummary =
                DataStreamUtils.mapPartition(
                        ratingData,
                        new MapPartitionFunction<
                                Ratings, Tuple5<Long, Long, Long, Integer, Integer>>() {
                            private static final long serialVersionUID = -3529850335007040435L;

                            @Override
                            public void mapPartition(
                                    Iterable<Ratings> values,
                                    Collector<Tuple5<Long, Long, Long, Integer, Integer>> out) {
                                long numUsers = 0L;
                                long numItems = 0L;
                                long numRatings = 0L;
                                int hottestUserPoint = 0;
                                int hottestItemPoint = 0;
                                for (Ratings ratings : values) {
                                    if (ratings.nodeId % 2L == 0L) {
                                        numUsers++;
                                        numRatings += ratings.scores.length;
                                        hottestUserPoint =
                                                Math.max(hottestUserPoint, ratings.numNeighbors);
                                    } else {
                                        numItems++;
                                        hottestItemPoint =
                                                Math.max(hottestItemPoint, ratings.numNeighbors);
                                    }
                                }
                                out.collect(
                                        Tuple5.of(
                                                numUsers,
                                                numItems,
                                                numRatings,
                                                hottestUserPoint,
                                                hottestItemPoint));
                            }
                        });

        return DataStreamUtils.reduce(
                        localSummary,
                        new ReduceFunction<Tuple5<Long, Long, Long, Integer, Integer>>() {
                            private static final long serialVersionUID = 3849683380245684843L;

                            @Override
                            public Tuple5<Long, Long, Long, Integer, Integer> reduce(
                                    Tuple5<Long, Long, Long, Integer, Integer> value1,
                                    Tuple5<Long, Long, Long, Integer, Integer> value2) {
                                value1.f0 += value2.f0;
                                value1.f1 += value2.f1;
                                value1.f2 += value2.f2;
                                value1.f3 = Math.max(value1.f3, value2.f3);
                                value1.f4 = Math.max(value1.f4, value2.f4);
                                return value1;
                            }
                        })
                .map(
                        new RichMapFunction<Tuple5<Long, Long, Long, Integer, Integer>, Ratings>() {
                            private static final long serialVersionUID = -2224348217053561771L;

                            @Override
                            public Ratings map(Tuple5<Long, Long, Long, Integer, Integer> value) {
                                Ratings profile = new Ratings();
                                profile.neighbors =
                                        new long[] {
                                            value.f0, value.f1, value.f2, value.f3, value.f4
                                        };
                                profile.scores = null;
                                return profile;
                            }
                        })
                .name("data_profiling");
    }

    private static class YtyAggregator implements PullStage.Aggregator<double[], double[]> {
        @Override
        public double[] add(double[] in, double[] acc) {

            if (acc == null) {
                acc = new double[in.length * in.length];
            }
            calcYty(in, acc);
            return acc;
        }

        @Override
        public double[] merge(double[] acc1, double[] acc2) {
            for (int i = 0; i < acc1.length; i++) {
                acc2[i] += acc1[i];
            }
            return acc2;
        }

        private void calcYty(double[] vec, double[] result) {
            for (int i = 0; i < vec.length; i++) {
                for (int j = 0; j < vec.length; j++) {
                    result[i * vec.length + j] += vec[i] * vec[j];
                }
            }
        }
    }

    private static double[][] sumYty(double[][] d1, double[][] d2) {
        Preconditions.checkArgument(d1[0].length == d2[0].length);
        for (int i = 0; i < d1[0].length; i++) {
            d2[0][i] += d1[0][i];
        }
        return d2;
    }

    /** The whole ratings of a user or an item. */
    public static class Ratings {

        public Ratings() {}

        /** Current node is a split node or not. */
        public boolean isSplit;

        /** Current node is a main node in split nodes or not. */
        public boolean isMainNode;

        /** UserId or itemId decided by identity. */
        public long nodeId;

        /** Number of neighbors. */
        public int numNeighbors;

        /** Neighbors of this nodeId. */
        public long[] neighbors;

        /** Scores from neighbors to this nodeId. */
        public double[] scores;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Als load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
