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

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

/** Runs a gradient boosting trees implementation. */
public class GBTRunner {

    /** Trains a gradient boosting tree model with given data and parameters. */
    static DataStream<GBTModelData> train(Table dataTable, GbtParams p) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();
        Tuple2<Table, DataStream<FeatureMeta>> preprocessResult =
                p.isInputVector
                        ? Preprocess.preprocessVecCol(dataTable, p)
                        : Preprocess.preprocessCols(dataTable, p);
        dataTable = preprocessResult.f0;
        DataStream<FeatureMeta> featureMeta = preprocessResult.f1;

        DataStream<Row> data = tEnv.toDataStream(dataTable);
        DataStream<Tuple2<Double, Long>> labelSumCount =
                DataStreamUtils.aggregate(data, new LabelSumCountFunction(p.labelCol));
        return boost(dataTable, p, featureMeta, labelSumCount);
    }

    private static DataStream<GBTModelData> boost(
            Table dataTable,
            GbtParams p,
            DataStream<FeatureMeta> featureMeta,
            DataStream<Tuple2<Double, Long>> labelSumCount) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();

        final String featureMetaBcName = "featureMeta";
        final String labelSumCountBcName = "labelSumCount";
        Map<String, DataStream<?>> bcMap = new HashMap<>();
        bcMap.put(featureMetaBcName, featureMeta);
        bcMap.put(labelSumCountBcName, labelSumCount);

        DataStream<LocalState> initStates =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(
                                tEnv.toDataStream(tEnv.fromValues(0), Integer.class)),
                        bcMap,
                        (inputs) -> {
                            //noinspection unchecked
                            DataStream<Integer> input = (DataStream<Integer>) (inputs.get(0));
                            return input.map(
                                    new InitLocalStateFunction(
                                            featureMetaBcName, labelSumCountBcName, p));
                        });

        DataStream<Row> data = tEnv.toDataStream(dataTable);
        final IterationID iterationID = new IterationID();
        DataStreamList dataStreamList =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(initStates.broadcast()),
                        ReplayableDataStreamList.notReplay(data, featureMeta),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(IterationConfig.OperatorLifeCycle.ALL_ROUND)
                                .build(),
                        new BoostIterationBody(iterationID, p));
        DataStream<LocalState> state = dataStreamList.get(0);
        return state.map(GBTModelData::fromLocalState);
    }

    private static class InitLocalStateFunction extends RichMapFunction<Integer, LocalState> {
        private final String featureMetaBcName;
        private final String labelSumCountBcName;
        private final GbtParams p;

        private InitLocalStateFunction(
                String featureMetaBcName, String labelSumCountBcName, GbtParams p) {
            this.featureMetaBcName = featureMetaBcName;
            this.labelSumCountBcName = labelSumCountBcName;
            this.p = p;
        }

        @Override
        public LocalState map(Integer value) {
            LocalState.Statics statics = new LocalState.Statics();
            statics.params = p;
            statics.featureMetas =
                    getRuntimeContext()
                            .<FeatureMeta>getBroadcastVariable(featureMetaBcName)
                            .toArray(new FeatureMeta[0]);
            if (!statics.params.isInputVector) {
                Arrays.sort(
                        statics.featureMetas,
                        Comparator.comparing(d -> ArrayUtils.indexOf(p.featureCols, d.name)));
            }
            statics.numFeatures = statics.featureMetas.length;
            statics.labelSumCount =
                    getRuntimeContext()
                            .<Tuple2<Double, Long>>getBroadcastVariable(labelSumCountBcName)
                            .get(0);
            return new LocalState(statics, new LocalState.Dynamics());
        }
    }

    private static class LabelSumCountFunction
            implements AggregateFunction<Row, Tuple2<Double, Long>, Tuple2<Double, Long>> {

        private final String labelCol;

        private LabelSumCountFunction(String labelCol) {
            this.labelCol = labelCol;
        }

        @Override
        public Tuple2<Double, Long> createAccumulator() {
            return Tuple2.of(0., 0L);
        }

        @Override
        public Tuple2<Double, Long> add(Row value, Tuple2<Double, Long> accumulator) {
            double label = ((Number) value.getFieldAs(labelCol)).doubleValue();
            return Tuple2.of(accumulator.f0 + label, accumulator.f1 + 1);
        }

        @Override
        public Tuple2<Double, Long> getResult(Tuple2<Double, Long> accumulator) {
            return accumulator;
        }

        @Override
        public Tuple2<Double, Long> merge(Tuple2<Double, Long> a, Tuple2<Double, Long> b) {
            return Tuple2.of(a.f0 + b.f0, a.f1 + b.f1);
        }
    }
}
