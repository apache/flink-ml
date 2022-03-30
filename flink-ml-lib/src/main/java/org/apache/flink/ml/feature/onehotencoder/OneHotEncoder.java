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

package org.apache.flink.ml.feature.onehotencoder;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * An Estimator which implements the one-hot encoding algorithm.
 *
 * <p>Data of selected input columns should be indexed numbers in order for OneHotEncoder to
 * function correctly.
 *
 * <p>See https://en.wikipedia.org/wiki/One-hot.
 */
public class OneHotEncoder
        implements Estimator<OneHotEncoder, OneHotEncoderModel>,
                OneHotEncoderParams<OneHotEncoder> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public OneHotEncoder() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OneHotEncoderModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkArgument(getHandleInvalid().equals(HasHandleInvalid.ERROR_INVALID));

        final String[] inputCols = getInputCols();

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Tuple2<Integer, Integer>> columnsAndValues =
                tEnv.toDataStream(inputs[0]).flatMap(new ExtractInputColsValueFunction(inputCols));

        DataStream<Tuple2<Integer, Integer>> modelData =
                DataStreamUtils.mapPartition(
                        columnsAndValues.keyBy(columnIdAndValue -> columnIdAndValue.f0),
                        new FindMaxIndexFunction());

        OneHotEncoderModel model =
                new OneHotEncoderModel().setModelData(tEnv.fromDataStream(modelData));
        ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static OneHotEncoder load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Extract values of input columns of input data.
     *
     * <p>Input: rows of input data containing designated input columns
     *
     * <p>Output: Pairs of column index and value stored in those columns
     */
    private static class ExtractInputColsValueFunction
            implements FlatMapFunction<Row, Tuple2<Integer, Integer>> {
        private final String[] inputCols;

        private ExtractInputColsValueFunction(String[] inputCols) {
            this.inputCols = inputCols;
        }

        @Override
        public void flatMap(Row row, Collector<Tuple2<Integer, Integer>> collector) {
            for (int i = 0; i < inputCols.length; i++) {
                Number number = (Number) row.getField(inputCols[i]);
                Preconditions.checkArgument(
                        number.intValue() == number.doubleValue(),
                        String.format("Value %s cannot be parsed as indexed integer.", number));
                Preconditions.checkArgument(
                        number.intValue() >= 0, "Negative value not supported.");
                collector.collect(new Tuple2<>(i, number.intValue()));
            }
        }
    }

    /** Function to find the max index value for each column. */
    private static class FindMaxIndexFunction
            implements MapPartitionFunction<Tuple2<Integer, Integer>, Tuple2<Integer, Integer>> {

        @Override
        public void mapPartition(
                Iterable<Tuple2<Integer, Integer>> iterable,
                Collector<Tuple2<Integer, Integer>> collector) {
            Map<Integer, Integer> map = new HashMap<>();
            for (Tuple2<Integer, Integer> value : iterable) {
                map.put(
                        value.f0,
                        Math.max(map.getOrDefault(value.f0, Integer.MIN_VALUE), value.f1));
            }
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                collector.collect(new Tuple2<>(entry.getKey(), entry.getValue()));
            }
        }
    }
}
