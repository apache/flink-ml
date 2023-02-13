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
import org.apache.flink.iteration.IterationID;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.operators.CacheDataCalcLocalHistsOperator;
import org.apache.flink.ml.common.gbt.operators.CalcLocalSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.HistogramAggregateFunction;
import org.apache.flink.ml.common.gbt.operators.PostSplitsOperator;
import org.apache.flink.ml.common.gbt.operators.SplitsAggregateFunction;
import org.apache.flink.ml.common.gbt.operators.TerminationOperator;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.types.Row;
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
        DataStream<TrainContext> trainContext = variableStreams.get(0);

        final String coLocationKey = "boosting";

        // In 1st round, cache all data. For all rounds calculate local histogram based on
        // current tree layer.
        SingleOutputStreamOperator<Histogram> localHists =
                data.connect(trainContext)
                        .transform(
                                "CacheDataCalcLocalHists",
                                TypeInformation.of(Histogram.class),
                                new CacheDataCalcLocalHistsOperator(gbtParams, iterationID));
        localHists.getTransformation().setCoLocationGroupKey("coLocationKey");

        DataStream<Histogram> globalHists = scatterReduceHistograms(localHists);

        SingleOutputStreamOperator<Splits> localSplits =
                globalHists.transform(
                        "CalcLocalSplits",
                        TypeInformation.of(Splits.class),
                        new CalcLocalSplitsOperator(iterationID));
        localHists.getTransformation().setCoLocationGroupKey(coLocationKey);
        DataStream<Splits> globalSplits =
                localSplits.broadcast().flatMap(new SplitsAggregateFunction());

        SingleOutputStreamOperator<Integer> updatedModelData =
                globalSplits
                        .broadcast()
                        .transform(
                                "PostSplits",
                                TypeInformation.of(Integer.class),
                                new PostSplitsOperator(iterationID));
        updatedModelData.getTransformation().setCoLocationGroupKey(coLocationKey);

        final OutputTag<GBTModelData> modelDataOutputTag =
                new OutputTag<>("model_data", TypeInformation.of(GBTModelData.class));
        SingleOutputStreamOperator<Integer> termination =
                updatedModelData.transform(
                        "check_termination",
                        Types.INT,
                        new TerminationOperator(iterationID, modelDataOutputTag));
        termination.getTransformation().setCoLocationGroupKey(coLocationKey);

        return new IterationBodyResult(
                DataStreamList.of(
                        updatedModelData.flatMap(
                                (d, out) -> {}, TypeInformation.of(TrainContext.class))),
                DataStreamList.of(termination.getSideOutput(modelDataOutputTag)),
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
