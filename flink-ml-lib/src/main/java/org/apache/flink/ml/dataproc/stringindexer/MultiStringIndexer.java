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

package org.apache.flink.ml.dataproc.stringindexer;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.core.Estimator;
import org.apache.flink.ml.common.EndOfStreamWindows;
import org.apache.flink.ml.param.Param;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import java.util.HashMap;
import java.util.Map;

/**
 * Encode several columns of strings to long type indices. The indices are consecutive long type
 * that start from 0. Each column is encoded separately.
 */
public class MultiStringIndexer implements Estimator<MultiStringIndexer,
        MultiStringIndexerModel>, MultiStringIndexerParams<MultiStringIndexer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    @Override
    public MultiStringIndexerModel fit(Table... inputs) {
        StreamTableEnvironment tEnv = (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);

        DataStream<Row> modelStream = input
                .flatMap(new FlattenFunction(getSelectedCols()))
                .keyBy(new DeduplicateKeySelector())
                .window(EndOfStreamWindows.get())
                .reduce(new DeduplicateFunction())
                .keyBy(new IndexKeySelector())
                .window(EndOfStreamWindows.get())
                .aggregate(new IndexFunction())
                .flatMap(new ModelDataSerializer());

        MultiStringIndexerModel op = new MultiStringIndexerModel();
        op.setSelectedCols(getSelectedCols())
                .setHandleInvalid(getHandleInvalid())
                .setOutputCols(getOutputCols())
                .setReservedCols(getReservedCols())
                .setModelData(tEnv.fromDataStream(modelStream));
        return op;
    }

    @Override
    public void save(String path) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Param<?>, Object> getUserDefinedParamMap() {
        return paramMap;
    }

    // converts each column into pairs of selected column index and column value.
    private static class FlattenFunction implements FlatMapFunction<Row, Tuple2<Integer, Object>> {
        private final String[] selectedCols;

        private FlattenFunction(String[] selectedCols) {
            this.selectedCols = selectedCols;
        }

        @Override
        public void flatMap(Row row, Collector<Tuple2<Integer, Object>> collector) {
            for (int i = 0; i < this.selectedCols.length; i++) {
                Object value = row.getField(this.selectedCols[i]);
                if (value != null) {
                    collector.collect(new Tuple2<>(i, value));
                }
            }
        }
    }

    private static class DeduplicateKeySelector implements KeySelector<Tuple2<Integer, Object>, Tuple2<Integer, Object>> {
        @Override
        public Tuple2<Integer, Object> getKey(Tuple2<Integer, Object> value) {
            return value;
        }
    }

    // removes duplicated value.
    private static class DeduplicateFunction implements ReduceFunction<Tuple2<Integer, Object>> {
        @Override
        public Tuple2<Integer, Object> reduce(Tuple2<Integer, Object> t0, Tuple2<Integer, Object> t1) {
            return t0;
        }
    }

    private static class IndexKeySelector implements KeySelector<Tuple2<Integer, Object>, Integer> {
        @Override
        public Integer getKey(Tuple2<Integer, Object> value) {
            return value.f0;
        }
    }

    // converts column value to unique index.
    private static class IndexFunction implements AggregateFunction<
            Tuple2<Integer, Object>,
            Tuple2<Integer, Map<Object, Long>>,
            Tuple2<Integer, Map<Object, Long>>> {
        private static final int DEFAULT_VALUE = -1;
        @Override
        public Tuple2<Integer, Map<Object, Long>> createAccumulator() {
            return new Tuple2<>(DEFAULT_VALUE, new HashMap<>());
        }

        @Override
        public Tuple2<Integer, Map<Object, Long>> add(
                Tuple2<Integer, Object> value,
                Tuple2<Integer, Map<Object, Long>> acc) {
            acc.f0 = value.f0;
            acc.f1.put(value.f1, (long) acc.f1.size());
            return acc;
        }

        @Override
        public Tuple2<Integer, Map<Object, Long>> getResult(Tuple2<Integer, Map<Object, Long>> acc) {
            return acc;
        }

        @Override
        public Tuple2<Integer, Map<Object, Long>> merge(
                Tuple2<Integer, Map<Object, Long>> acc0,
                Tuple2<Integer, Map<Object, Long>> acc1) {
            if (acc0.f0.equals(DEFAULT_VALUE)) {
                acc0.f0 = acc1.f0;
            }
            for (Map.Entry<Object, Long> entry: acc1.f1.entrySet()) {
                acc0.f1.put(entry.getKey(), (long) acc0.f1.size());
            }
            return acc0;
        }
    }

    private static class ModelDataSerializer implements FlatMapFunction<Tuple2<Integer, Map<Object, Long>>, Row> {
        @Override
        public void flatMap(Tuple2<Integer, Map<Object, Long>> value, Collector<Row> collector) {
            for (Map.Entry<Object, Long> entry: value.f1.entrySet()) {
                collector.collect(Row.of(value.f0, entry.getKey(), entry.getValue()));
            }
        }
    }
}
