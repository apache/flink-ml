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

package org.apache.flink.ml.feature.imputer;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.util.QuantileSummary;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.FlinkRuntimeException;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * The imputer for completing missing values of the input columns.
 *
 * <p>Missing values can be imputed using the statistics(mean, median or most frequent) of each
 * column in which the missing values are located. The input columns should be of numeric type.
 *
 * <p>Note that the mean/median/most_frequent value is computed after filtering out missing values.
 * All null values in the input columns are also treated as missing, and so are imputed.
 */
public class Imputer implements Estimator<Imputer, ImputerModel>, ImputerParams<Imputer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Imputer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public ImputerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkArgument(
                getInputCols().length == getOutputCols().length,
                "Num of input columns and output columns are inconsistent.");
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> inputData = tEnv.toDataStream(inputs[0]);

        DataStream<ImputerModelData> modelData;
        switch (getStrategy()) {
            case MEAN:
                modelData =
                        DataStreamUtils.aggregate(
                                inputData,
                                new MeanStrategyAggregator(getInputCols(), getMissingValue()),
                                Types.MAP(Types.STRING, Types.TUPLE(Types.DOUBLE, Types.LONG)),
                                ImputerModelData.TYPE_INFO);
                break;
            case MEDIAN:
                modelData =
                        DataStreamUtils.aggregate(
                                inputData,
                                new MedianStrategyAggregator(
                                        getInputCols(), getMissingValue(), getRelativeError()),
                                Types.MAP(Types.STRING, TypeInformation.of(QuantileSummary.class)),
                                ImputerModelData.TYPE_INFO);
                break;
            case MOST_FREQUENT:
                modelData =
                        DataStreamUtils.aggregate(
                                inputData,
                                new MostFrequentStrategyAggregator(
                                        getInputCols(), getMissingValue()),
                                Types.MAP(Types.STRING, Types.MAP(Types.DOUBLE, Types.LONG)),
                                ImputerModelData.TYPE_INFO);
                break;
            default:
                throw new RuntimeException("Unsupported strategy of Imputer: " + getStrategy());
        }

        Schema schema =
                Schema.newBuilder()
                        .column("surrogates", DataTypes.MAP(DataTypes.STRING(), DataTypes.DOUBLE()))
                        .build();
        ImputerModel model =
                new ImputerModel().setModelData(tEnv.fromDataStream(modelData, schema));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    /**
     * A stream operator to compute the mean value of all input columns of the input bounded data
     * stream.
     */
    private static class MeanStrategyAggregator
            implements AggregateFunction<Row, Map<String, Tuple2<Double, Long>>, ImputerModelData> {

        private final String[] columnNames;
        private final double missingValue;

        public MeanStrategyAggregator(String[] columnNames, double missingValue) {
            this.columnNames = columnNames;
            this.missingValue = missingValue;
        }

        @Override
        public Map<String, Tuple2<Double, Long>> createAccumulator() {
            Map<String, Tuple2<Double, Long>> accumulators = new HashMap<>();
            Arrays.stream(columnNames).forEach(x -> accumulators.put(x, Tuple2.of(0.0, 0L)));
            return accumulators;
        }

        @Override
        public Map<String, Tuple2<Double, Long>> add(
                Row row, Map<String, Tuple2<Double, Long>> accumulators) {
            accumulators.forEach(
                    (col, sumAndNum) -> {
                        Object rawValue = row.getField(col);
                        if (rawValue != null) {
                            Double value = Double.valueOf(rawValue.toString());
                            if (!value.equals(missingValue) && !value.equals(Double.NaN)) {
                                sumAndNum.f0 += value;
                                sumAndNum.f1 += 1;
                            }
                        }
                    });
            return accumulators;
        }

        @Override
        public ImputerModelData getResult(Map<String, Tuple2<Double, Long>> map) {
            long numRows = map.entrySet().stream().findFirst().get().getValue().f1;
            Preconditions.checkState(
                    numRows > 0, "The training set is empty or does not contains valid data.");

            Map<String, Double> surrogates = new HashMap<>();
            map.forEach((col, sumAndNum) -> surrogates.put(col, sumAndNum.f0 / sumAndNum.f1));
            return new ImputerModelData(surrogates);
        }

        @Override
        public Map<String, Tuple2<Double, Long>> merge(
                Map<String, Tuple2<Double, Long>> acc1, Map<String, Tuple2<Double, Long>> acc2) {
            Preconditions.checkArgument(acc1.size() == acc2.size());

            acc1.forEach(
                    (col, numAndSum) -> {
                        acc2.get(col).f0 += numAndSum.f0;
                        acc2.get(col).f1 += numAndSum.f1;
                    });
            return acc2;
        }
    }

    /**
     * A stream operator to compute the median value of all input columns of the input bounded data
     * stream.
     */
    private static class MedianStrategyAggregator
            implements AggregateFunction<Row, Map<String, QuantileSummary>, ImputerModelData> {
        private final String[] columnNames;
        private final double missingValue;
        private final double relativeError;

        public MedianStrategyAggregator(
                String[] columnNames, double missingValue, double relativeError) {
            this.columnNames = columnNames;
            this.missingValue = missingValue;
            this.relativeError = relativeError;
        }

        @Override
        public Map<String, QuantileSummary> createAccumulator() {
            Map<String, QuantileSummary> summaries = new HashMap<>();
            Arrays.stream(columnNames)
                    .forEach(x -> summaries.put(x, new QuantileSummary(relativeError)));
            return summaries;
        }

        @Override
        public Map<String, QuantileSummary> add(Row row, Map<String, QuantileSummary> summaries) {
            summaries.forEach(
                    (col, summary) -> {
                        Object rawValue = row.getField(col);
                        if (rawValue != null) {
                            Double value = Double.valueOf(rawValue.toString());
                            if (!value.equals(missingValue) && !value.equals(Double.NaN)) {
                                summary.insert(value);
                            }
                        }
                    });
            return summaries;
        }

        @Override
        public ImputerModelData getResult(Map<String, QuantileSummary> summaries) {
            Map<String, Double> surrogates = new HashMap<>();
            summaries.forEach(
                    (col, summary) -> {
                        QuantileSummary compressed = summary.compress();
                        if (compressed.isEmpty()) {
                            throw new FlinkRuntimeException(
                                    String.format(
                                            "Surrogate cannot be computed. All the values in column [%s] are null, NaN or missingValue.",
                                            col));
                        }
                        double median = compressed.query(0.5);
                        surrogates.put(col, median);
                    });
            return new ImputerModelData(surrogates);
        }

        @Override
        public Map<String, QuantileSummary> merge(
                Map<String, QuantileSummary> acc1, Map<String, QuantileSummary> acc2) {
            Preconditions.checkArgument(acc1.size() == acc2.size());

            acc1.forEach(
                    (col, summary1) -> {
                        QuantileSummary summary2 = acc2.get(col).compress();
                        acc2.put(col, summary2.merge(summary1.compress()));
                    });
            return acc2;
        }
    }

    /**
     * A stream operator to compute the most frequent value of all input columns of the input
     * bounded data stream.
     */
    private static class MostFrequentStrategyAggregator
            implements AggregateFunction<Row, Map<String, Map<Double, Long>>, ImputerModelData> {
        private final String[] columnNames;
        private final double missingValue;

        public MostFrequentStrategyAggregator(String[] columnNames, double missingValue) {
            this.columnNames = columnNames;
            this.missingValue = missingValue;
        }

        @Override
        public Map<String, Map<Double, Long>> createAccumulator() {
            Map<String, Map<Double, Long>> accumulators = new HashMap<>();
            Arrays.stream(columnNames).forEach(x -> accumulators.put(x, new HashMap<>()));
            return accumulators;
        }

        @Override
        public Map<String, Map<Double, Long>> add(
                Row row, Map<String, Map<Double, Long>> accumulators) {
            accumulators.forEach(
                    (col, counts) -> {
                        Object rawValue = row.getField(col);
                        if (rawValue != null) {
                            Double value = Double.valueOf(rawValue.toString());
                            if (!value.equals(missingValue) && !value.equals(Double.NaN)) {
                                if (counts.containsKey(value)) {
                                    counts.put(value, counts.get(value) + 1);
                                } else {
                                    counts.put(value, 1L);
                                }
                            }
                        }
                    });
            return accumulators;
        }

        @Override
        public ImputerModelData getResult(Map<String, Map<Double, Long>> map) {
            long validColumns =
                    map.entrySet().stream().filter(x -> x.getValue().size() > 0).count();
            Preconditions.checkState(
                    validColumns > 0, "The training set is empty or does not contains valid data.");

            Map<String, Double> surrogates = new HashMap<>();
            map.forEach(
                    (col, counts) -> {
                        long maxCnt = Long.MIN_VALUE;
                        double value = Double.NaN;
                        for (Map.Entry<Double, Long> entry : counts.entrySet()) {
                            if (maxCnt <= entry.getValue()) {
                                value =
                                        maxCnt == entry.getValue()
                                                ? Math.min(entry.getKey(), value)
                                                : entry.getKey();
                                maxCnt = entry.getValue();
                            }
                        }
                        surrogates.put(col, value);
                    });
            return new ImputerModelData(surrogates);
        }

        @Override
        public Map<String, Map<Double, Long>> merge(
                Map<String, Map<Double, Long>> acc1, Map<String, Map<Double, Long>> acc2) {
            Preconditions.checkArgument(acc1.size() == acc2.size());

            acc1.forEach(
                    (col, counts) -> {
                        Map<Double, Long> map = acc2.get(col);
                        counts.forEach(
                                (value, cnt) -> {
                                    if (map.containsKey(value)) {
                                        map.put(value, cnt + map.get(value));
                                    } else {
                                        map.put(value, cnt);
                                    }
                                });
                    });
            return acc2;
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Imputer load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
