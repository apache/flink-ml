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
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.operators.CacheDataCalcLocalHistsOperator;
import org.apache.flink.ml.common.gbt.operators.CalcLocalSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.HistogramAggregateFunction;
import org.apache.flink.ml.common.gbt.operators.PostSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.SharedStorageConstants;
import org.apache.flink.ml.common.gbt.operators.SplitsAggregateFunction;
import org.apache.flink.ml.common.gbt.operators.TerminationOperator;
import org.apache.flink.ml.common.sharedstorage.ItemDescriptor;
import org.apache.flink.ml.common.sharedstorage.SharedStorageBody;
import org.apache.flink.ml.common.sharedstorage.SharedStorageUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.types.Row;
import org.apache.flink.util.OutputTag;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implements iteration body for boosting algorithms. This implementation uses horizontal partition
 * of data and row-store storage of instances.
 */
class BoostIterationBody implements IterationBody {
    private final BoostingStrategy strategy;

    public BoostIterationBody(BoostingStrategy strategy) {
        this.strategy = strategy;
    }

    private SharedStorageBody.SharedStorageBodyResult sharedStorageBody(
            List<DataStream<?>> inputs) {
        //noinspection unchecked
        DataStream<Row> data = (DataStream<Row>) inputs.get(0);
        //noinspection unchecked
        DataStream<TrainContext> trainContext = (DataStream<TrainContext>) inputs.get(1);

        Map<ItemDescriptor<?>, String> ownerMap = new HashMap<>();

        // In 1st round, cache all data. For all rounds calculate local histogram based on
        // current tree layer.
        CacheDataCalcLocalHistsOperator cacheDataCalcLocalHistsOp =
                new CacheDataCalcLocalHistsOperator(strategy);
        SingleOutputStreamOperator<Histogram> localHists =
                data.connect(trainContext)
                        .transform(
                                "CacheDataCalcLocalHists",
                                TypeInformation.of(Histogram.class),
                                cacheDataCalcLocalHistsOp);
        for (ItemDescriptor<?> s : SharedStorageConstants.OWNED_BY_CACHE_DATA_CALC_LOCAL_HISTS_OP) {
            ownerMap.put(s, cacheDataCalcLocalHistsOp.getSharedStorageAccessorID());
        }

        DataStream<Histogram> globalHists = scatterReduceHistograms(localHists);

        SingleOutputStreamOperator<Splits> localSplits =
                globalHists.transform(
                        "CalcLocalSplits",
                        TypeInformation.of(Splits.class),
                        new CalcLocalSplitsOperator());
        DataStream<Splits> globalSplits =
                localSplits.broadcast().flatMap(new SplitsAggregateFunction());

        PostSplitsOperator postSplitsOp = new PostSplitsOperator();
        SingleOutputStreamOperator<Integer> updatedModelData =
                globalSplits
                        .broadcast()
                        .transform("PostSplits", TypeInformation.of(Integer.class), postSplitsOp);
        for (ItemDescriptor<?> descriptor : SharedStorageConstants.OWNED_BY_POST_SPLITS_OP) {
            ownerMap.put(descriptor, postSplitsOp.getSharedStorageAccessorID());
        }

        final OutputTag<GBTModelData> finalModelDataOutputTag =
                new OutputTag<>("model_data", TypeInformation.of(GBTModelData.class));
        SingleOutputStreamOperator<Integer> termination =
                updatedModelData.transform(
                        "CheckTermination",
                        Types.INT,
                        new TerminationOperator(finalModelDataOutputTag));
        DataStream<GBTModelData> finalModelData =
                termination.getSideOutput(finalModelDataOutputTag);

        return new SharedStorageBody.SharedStorageBodyResult(
                Arrays.asList(updatedModelData, finalModelData, termination),
                Arrays.asList(localHists, localSplits, updatedModelData, termination),
                ownerMap);
    }

    @Override
    public IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams) {
        DataStream<Row> data = dataStreams.get(0);
        DataStream<TrainContext> trainContext = variableStreams.get(0);

        List<DataStream<?>> outputs =
                SharedStorageUtils.withSharedStorage(
                        Arrays.asList(data, trainContext), this::sharedStorageBody);

        DataStream<?> updatedModelData = outputs.get(0);
        DataStream<?> finalModelData = outputs.get(1);
        DataStream<?> termination = outputs.get(2);
        return new IterationBodyResult(
                DataStreamList.of(
                        updatedModelData.flatMap(
                                (d, out) -> {}, TypeInformation.of(TrainContext.class))),
                DataStreamList.of(finalModelData),
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
                            public Integer getKey(Tuple2<Integer, Histogram> value) {
                                return value.f0;
                            }
                        })
                .map(d -> d.f1)
                .flatMap(new HistogramAggregateFunction());
    }
}
