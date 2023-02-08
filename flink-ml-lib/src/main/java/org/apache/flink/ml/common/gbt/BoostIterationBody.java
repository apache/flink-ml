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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.Partitioner;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.operators.CacheDataCalcLocalHistsOperator;
import org.apache.flink.ml.common.gbt.operators.CalcLocalSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.HistogramAggregateFunction;
import org.apache.flink.ml.common.gbt.operators.PostSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.SplitsAggregateFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import org.apache.commons.lang3.ArrayUtils;

/**
 * Implements iteration body for boosting algorithms. This implementation uses horizontal partition
 * of data and row-store storage of instances.
 */
class BoostIterationBody implements IterationBody {
    private final IterationID iterationID;
    private final GbtParams gbtParams;

    public BoostIterationBody(IterationID iterationID, GbtParams gbtParams) {
        this.iterationID = iterationID;
        this.gbtParams = gbtParams;
    }

    @Override
    public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
        DataStream<Row> data = dataStreams.get(0);
        DataStream<LocalState> localState = variableStreams.get(0);

        final OutputTag<LocalState> stateOutputTag =
                new OutputTag<>("state", TypeInformation.of(LocalState.class));

        final OutputTag<LocalState> finalStateOutputTag =
                new OutputTag<>("final_state", TypeInformation.of(LocalState.class));

        /**
         * In the iteration, some data needs to be shared between subtasks of different operators
         * within one machine. We use {@link IterationSharedStorage} with co-location mechanism to
         * achieve such purpose. The data is stored in JVM static region, and is accessed through
         * string keys from different operator subtasks. Note the first operator to put the data is
         * the owner of the data, and only the owner can update or delete the data.
         *
         * <p>To be specified, in gradient boosting trees algorithm, there three types of shared
         * data:
         *
         * <ul>
         *   <li>Instances (after binned) and their corresponding predictions, gradients, and
         *       hessians are shared to avoid being stored multiple times or communication.
         *   <li>When initializing every new tree, instances need to be shuffled and split to
         *       bagging instances and non-bagging ones. To reduce the cost, we shuffle instance
         *       indices other than instances. Therefore, the shuffle indices need to be shared to
         *       access actual instances.
         *   <li>After splitting nodes of each layer, instance indices need to be swapped to
         *       maintain {@link LearningNode#slice} and {@link LearningNode#oob}. However, we
         *       cannot directly update the data of shuffle indices above, as it already has an
         *       owner. So we use another key to store instance indices after swapping.
         * </ul>
         */
        final String sharedInstancesKey = "instances";
        final String sharedPredGradHessKey = "preds_grads_hessians";
        final String sharedShuffledIndicesKey = "shuffled_indices";
        final String sharedSwappedIndicesKey = "swapped_indices";

        final String coLocationKey = "boosting";

        // In 1st round, cache all data. For all rounds calculate local histogram based on
        // current tree layer.
        SingleOutputStreamOperator<Histogram> localHists =
                data.connect(localState)
                        .transform(
                                "CacheDataCalcLocalHists",
                                TypeInformation.of(Histogram.class),
                                new CacheDataCalcLocalHistsOperator(
                                        gbtParams,
                                        iterationID,
                                        sharedInstancesKey,
                                        sharedPredGradHessKey,
                                        sharedShuffledIndicesKey,
                                        sharedSwappedIndicesKey,
                                        stateOutputTag));
        localHists.getTransformation().setCoLocationGroupKey("coLocationKey");
        DataStream<LocalState> modelData = localHists.getSideOutput(stateOutputTag);

        DataStream<Histogram> globalHists = scatterReduceHistograms(localHists);

        SingleOutputStreamOperator<Splits> localSplits =
                modelData
                        .connect(globalHists)
                        .transform(
                                "CalcLocalSplits",
                                TypeInformation.of(Splits.class),
                                new CalcLocalSplitsOperator(stateOutputTag));
        localHists.getTransformation().setCoLocationGroupKey(coLocationKey);
        DataStream<Splits> globalSplits =
                localSplits.broadcast().flatMap(new SplitsAggregateFunction());

        SingleOutputStreamOperator<LocalState> updatedModelData =
                modelData
                        .connect(globalSplits.broadcast())
                        .transform(
                                "PostSplits",
                                TypeInformation.of(LocalState.class),
                                new PostSplitsOperator(
                                        iterationID,
                                        sharedInstancesKey,
                                        sharedPredGradHessKey,
                                        sharedShuffledIndicesKey,
                                        sharedSwappedIndicesKey,
                                        finalStateOutputTag));
        updatedModelData.getTransformation().setCoLocationGroupKey(coLocationKey);

        DataStream<Integer> termination =
                updatedModelData.flatMap(
                        new FlatMapFunction<LocalState, Integer>() {
                            @Override
                            public void flatMap(LocalState value, Collector<Integer> out) {
                                LocalState.Dynamics dynamics = value.dynamics;
                                boolean terminated =
                                        !dynamics.inWeakLearner
                                                && dynamics.roots.size()
                                                        == value.statics.params.maxIter;
                                // TODO: add validation error rate
                                if (!terminated) {
                                    out.collect(0);
                                }
                            }
                        });
        termination.getTransformation().setCoLocationGroupKey(coLocationKey);

        return new IterationBodyResult(
                DataStreamList.of(updatedModelData),
                DataStreamList.of(updatedModelData.getSideOutput(finalStateOutputTag)),
                termination);
    }

    public DataStream<Histogram> scatterReduceHistograms(DataStream<Histogram> localHists) {
        return localHists
                .flatMap(
                        (FlatMapFunction<Histogram, Tuple2<Integer, Histogram>>)
                                (value, out) -> {
                                    double[] hists = value.hists;
                                    int[] recvcnts = value.recvcnts;
                                    int p = 0;
                                    for (int i = 0; i < recvcnts.length; i += 1) {
                                        out.collect(
                                                Tuple2.of(
                                                        i,
                                                        new Histogram(
                                                                value.subtaskId,
                                                                ArrayUtils.subarray(
                                                                        hists, p, p + recvcnts[i]),
                                                                recvcnts)));
                                        p += recvcnts[i];
                                    }
                                })
                .returns(new TypeHint<Tuple2<Integer, Histogram>>() {})
                .partitionCustom(
                        new Partitioner<Integer>() {
                            @Override
                            public int partition(Integer key, int numPartitions) {
                                return key;
                            }
                        },
                        new KeySelector<Tuple2<Integer, Histogram>, Integer>() {
                            @Override
                            public Integer getKey(Tuple2<Integer, Histogram> value)
                                    throws Exception {
                                return value.f0;
                            }
                        })
                .map(d -> d.f1)
                .flatMap(new HistogramAggregateFunction());
    }
}
