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

package org.apache.flink.ml.stats.chisqtest;

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Chi-square test of independence of variables in a contingency table. This Transformer computes
 * the chi-square statistic,p-value,and dof(number of degrees of freedom) for every feature in the
 * contingency table, which constructed from the `observed` for each categorical values. All label
 * and feature values must be categorical.
 *
 * <p>See: http://en.wikipedia.org/wiki/Chi-squared_test.
 */
public class ChiSqTest implements AlgoOperator<ChiSqTest>, ChiSqTestParams<ChiSqTest> {

    final String bcCategoricalMarginsKey = "bcCategoricalMarginsKey";
    final String bcLabelMarginsKey = "bcLabelMarginsKey";

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public ChiSqTest() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {

        final String[] inputCols = getInputCols();
        String labelCol = getLabelCol();

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        SingleOutputStreamOperator<Tuple3<String, Object, Object>> colAndFeatureAndLabel =
                tEnv.toDataStream(inputs[0])
                        .flatMap(new ExtractColAndFeatureAndLabel(inputCols, labelCol));

        // compute the observed frequencies
        DataStream<Tuple4<String, Object, Object, Long>> observedFreq =
                DataStreamUtils.mapPartition(
                        colAndFeatureAndLabel.keyBy(Tuple3::hashCode),
                        new GenerateObservedFrequencies());

        SingleOutputStreamOperator<Tuple4<String, Object, Object, Long>> filledObservedFreq =
                observedFreq
                        .transform(
                                "filledObservedFreq",
                                Types.TUPLE(
                                        Types.STRING,
                                        Types.GENERIC(Object.class),
                                        Types.GENERIC(Object.class),
                                        Types.LONG),
                                new FillZeroFunc())
                        .setParallelism(1);

        // return a DataStream of the marginal sums of the factors
        DataStream<Tuple3<String, Object, Long>> categoricalMargins =
                DataStreamUtils.mapPartition(
                        observedFreq.keyBy(tuple -> new Tuple2<>(tuple.f0, tuple.f1).hashCode()),
                        new MapPartitionFunction<
                                Tuple4<String, Object, Object, Long>,
                                Tuple3<String, Object, Long>>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple4<String, Object, Object, Long>> iterable,
                                    Collector<Tuple3<String, Object, Long>> out) {
                                HashMap<Tuple2<String, Object>, Long> map = new HashMap<>();

                                for (Tuple4<String, Object, Object, Long> tuple : iterable) {
                                    Long observedFreq = tuple.f3;
                                    Tuple2<String, Object> key = new Tuple2<>(tuple.f0, tuple.f1);

                                    if (map.containsKey(key)) {
                                        Long count = map.get(key);
                                        map.put(key, count + observedFreq);
                                    } else {
                                        map.put(key, observedFreq);
                                    }
                                }

                                for (Tuple2<String, Object> key : map.keySet()) {
                                    Long categoricalMargin = map.get(key);
                                    out.collect(new Tuple3<>(key.f0, key.f1, categoricalMargin));
                                }
                            }
                        });

        // return a DataStream of the marginal sums of the labels
        DataStream<Tuple3<String, Object, Long>> labelMargins =
                DataStreamUtils.mapPartition(
                        observedFreq.keyBy(tuple -> new Tuple2<>(tuple.f0, tuple.f2).hashCode()),
                        new MapPartitionFunction<
                                Tuple4<String, Object, Object, Long>,
                                Tuple3<String, Object, Long>>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple4<String, Object, Object, Long>> iterable,
                                    Collector<Tuple3<String, Object, Long>> out) {

                                HashMap<Tuple2<String, Object>, Long> map = new HashMap<>();

                                for (Tuple4<String, Object, Object, Long>
                                        colAndFeatureAndLabelAndCount : iterable) {
                                    Long observedFreq = colAndFeatureAndLabelAndCount.f3;
                                    Tuple2<String, Object> key =
                                            new Tuple2<>(
                                                    colAndFeatureAndLabelAndCount.f0,
                                                    colAndFeatureAndLabelAndCount.f2);

                                    if (map.containsKey(key)) {
                                        Long count = map.get(key);
                                        map.put(key, count + observedFreq);
                                    } else {
                                        map.put(key, observedFreq);
                                    }
                                }

                                for (Tuple2<String, Object> key : map.keySet()) {
                                    Long labelMargin = map.get(key);
                                    out.collect(new Tuple3<>(key.f0, key.f1, labelMargin));
                                }
                            }
                        });

        Function<List<DataStream<?>>, DataStream<Tuple3<String, Double, Integer>>> function =
                dataStreams -> {
                    DataStream stream = dataStreams.get(0);
                    return stream.map(new ChiSqFunc(bcCategoricalMarginsKey, bcLabelMarginsKey));
                };

        HashMap<String, DataStream<?>> bcMap =
                new HashMap<String, DataStream<?>>() {
                    {
                        put(bcCategoricalMarginsKey, categoricalMargins);
                        put(bcLabelMarginsKey, labelMargins);
                    }
                };

        DataStream<Tuple3<String, Double, Integer>> categoricalStatistics =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(filledObservedFreq), bcMap, function);

        SingleOutputStreamOperator<Row> chiSqTestResult =
                categoricalStatistics
                        .transform(
                                "chiSqTestResult",
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.STRING, Types.DOUBLE, Types.DOUBLE, Types.INT
                                        },
                                        new String[] {
                                            "column", "pValue", "statistic", "degreesOfFreedom"
                                        }),
                                new AggregateChiSqFunc())
                        .setParallelism(1);

        return new Table[] {tEnv.fromDataStream(chiSqTestResult)};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static ChiSqTest load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class ExtractColAndFeatureAndLabel
            extends RichFlatMapFunction<Row, Tuple3<String, Object, Object>> {
        private final String[] inputCols;
        private final String labelCol;

        public ExtractColAndFeatureAndLabel(String[] inputCols, String labelCol) {
            this.inputCols = inputCols;
            this.labelCol = labelCol;
        }

        @Override
        public void flatMap(Row row, Collector<Tuple3<String, Object, Object>> collector) {

            Object label = row.getFieldAs(labelCol);

            for (String colName : inputCols) {
                Object value = row.getField(colName);
                collector.collect(new Tuple3<>(colName, value, label));
            }
        }
    }

    /**
     * Compute a frequency table(DataStream) of the factors(categorical values). The returned
     * DataStream contains the observed frequencies (i.e. number of occurrences) in each category.
     */
    private static class GenerateObservedFrequencies
            extends RichMapPartitionFunction<
                    Tuple3<String, Object, Object>, Tuple4<String, Object, Object, Long>> {

        @Override
        public void mapPartition(
                Iterable<Tuple3<String, Object, Object>> iterable,
                Collector<Tuple4<String, Object, Object, Long>> out) {

            HashMap<Tuple3<String, Object, Object>, Long> map = new HashMap<>();

            for (Tuple3<String, Object, Object> key : iterable) {
                if (map.containsKey(key)) {
                    Long count = map.get(key);
                    map.put(key, count + 1);
                } else {
                    map.put(key, 1L);
                }
            }

            for (Tuple3<String, Object, Object> key : map.keySet()) {
                Long count = map.get(key);
                out.collect(new Tuple4<>(key.f0, key.f1, key.f2, count));
            }
        }
    }

    /** Fill the factors which frequencies are zero in frequency table. */
    private static class FillZeroFunc
            extends AbstractStreamOperator<Tuple4<String, Object, Object, Long>>
            implements OneInputStreamOperator<
                            Tuple4<String, Object, Object, Long>,
                            Tuple4<String, Object, Object, Long>>,
                    BoundedOneInput {

        HashMap<Tuple2<String, Object>, ArrayList<Tuple2<Object, Long>>> values = new HashMap<>();
        HashSet<Object> numLabels = new HashSet<>();

        @Override
        public void endInput() {

            for (Map.Entry<Tuple2<String, Object>, ArrayList<Tuple2<Object, Long>>> entry :
                    values.entrySet()) {
                ArrayList<Tuple2<Object, Long>> labelAndCountList = entry.getValue();
                Tuple2<String, Object> categoricalKey = entry.getKey();

                List<Object> existingLabels =
                        labelAndCountList.stream().map(v -> v.f0).collect(Collectors.toList());

                for (Object label : numLabels) {
                    if (!existingLabels.contains(label)) {
                        Tuple2<Object, Long> generatedLabelCount = new Tuple2<>(label, 0L);
                        labelAndCountList.add(generatedLabelCount);
                    }
                }

                for (Tuple2<Object, Long> labelAndCount : labelAndCountList) {
                    output.collect(
                            new StreamRecord<>(
                                    new Tuple4<>(
                                            categoricalKey.f0,
                                            categoricalKey.f1,
                                            labelAndCount.f0,
                                            labelAndCount.f1)));
                }
            }
        }

        @Override
        public void processElement(StreamRecord<Tuple4<String, Object, Object, Long>> element) {
            Tuple4<String, Object, Object, Long> colAndCategoryAndLabelAndCount =
                    element.getValue();
            Tuple2<String, Object> key =
                    new Tuple2<>(
                            colAndCategoryAndLabelAndCount.f0, colAndCategoryAndLabelAndCount.f1);
            Tuple2<Object, Long> labelAndCount =
                    new Tuple2<>(
                            colAndCategoryAndLabelAndCount.f2, colAndCategoryAndLabelAndCount.f3);
            ArrayList<Tuple2<Object, Long>> labelAndCountList = values.get(key);

            if (labelAndCountList == null) {
                ArrayList<Tuple2<Object, Long>> value = new ArrayList<>();
                value.add(labelAndCount);
                values.put(key, value);
            } else {
                labelAndCountList.add(labelAndCount);
            }

            numLabels.add(colAndCategoryAndLabelAndCount.f2);
        }
    }

    /**
     * Conduct Pearson's independence test in a contingency table that constructed from the input
     * `observed` for each categorical values.
     */
    private static class ChiSqFunc
            extends RichMapFunction<
                    Tuple4<String, Object, Object, Long>, Tuple3<String, Double, Integer>> {

        private final String bcCategoricalMarginsKey;
        private final String bcLabelMarginsKey;
        private final Map<Tuple2<String, Object>, Long> categoricalMargins = new HashMap<>();
        private final Map<Tuple2<String, Object>, Long> labelMargins = new HashMap<>();

        double sampleSize = 0;
        int numLabels = 0;
        HashMap<String, Integer> col2NumCategories = new HashMap<>();

        public ChiSqFunc(String bcCategoricalMarginsKey, String bcLabelMarginsKey) {
            this.bcCategoricalMarginsKey = bcCategoricalMarginsKey;
            this.bcLabelMarginsKey = bcLabelMarginsKey;
        }

        @Override
        public Tuple3<String, Double, Integer> map(Tuple4<String, Object, Object, Long> v) {
            if (categoricalMargins.isEmpty()) {
                List<Tuple3<String, Object, Long>> categoricalMarginList =
                        getRuntimeContext().getBroadcastVariable(bcCategoricalMarginsKey);
                List<Tuple3<String, Object, Long>> labelMarginList =
                        getRuntimeContext().getBroadcastVariable(bcLabelMarginsKey);

                for (Tuple3<String, Object, Long> colAndFeatureAndCount : categoricalMarginList) {
                    String theColName = colAndFeatureAndCount.f0;
                    col2NumCategories.merge(theColName, 1, Integer::sum);
                }

                numLabels = (int) labelMarginList.stream().map(x -> x.f1).distinct().count();

                for (Tuple3<String, Object, Long> colAndFeatureAndCount : categoricalMarginList) {
                    categoricalMargins.put(
                            new Tuple2<>(colAndFeatureAndCount.f0, colAndFeatureAndCount.f1),
                            colAndFeatureAndCount.f2);
                }

                Map<String, Double> sampleSizeCount = new HashMap<>();
                String tmpKey = null;

                for (Tuple3<String, Object, Long> colAndLabelAndCount : labelMarginList) {
                    String col = colAndLabelAndCount.f0;

                    if (tmpKey == null) {
                        tmpKey = col;
                        sampleSizeCount.put(col, 0D);
                    }

                    sampleSizeCount.computeIfPresent(
                            col, (k, count) -> count + colAndLabelAndCount.f2);
                    labelMargins.put(
                            new Tuple2<>(col, colAndLabelAndCount.f1), colAndLabelAndCount.f2);
                }

                Optional<Double> sampleSizeOpt =
                        sampleSizeCount.values().stream().reduce(Double::sum);
                Preconditions.checkArgument(sampleSizeOpt.isPresent());
                sampleSize = sampleSizeOpt.get();
            }

            String colName = v.f0;
            // Degrees of freedom
            int dof = (col2NumCategories.get(colName) - 1) * (numLabels - 1);

            Tuple2<String, Object> category = new Tuple2<>(v.f0, v.f1);

            Tuple2<String, Object> colAndLabelKey = new Tuple2<>(v.f0, v.f2);
            Long theCategoricalMargin = categoricalMargins.get(category);
            Long theLabelMargin = labelMargins.get(colAndLabelKey);
            Long observed = v.f3;

            double expected = (double) (theLabelMargin * theCategoricalMargin) / sampleSize;
            double categoricalStatistic = pearsonFunc(observed, expected);

            return new Tuple3<>(colName, categoricalStatistic, dof);
        }

        // Pearson's chi-squared test: http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
        private double pearsonFunc(double observed, double expected) {
            double dev = observed - expected;
            return dev * dev / expected;
        }
    }

    /**
     * This function computes the Pearson's chi-squared statistic , p-value ,and the number of
     * degrees of freedom for every feature across the input DataStream.
     */
    private static class AggregateChiSqFunc extends AbstractStreamOperator<Row>
            implements OneInputStreamOperator<Tuple3<String, Double, Integer>, Row>,
                    BoundedOneInput {

        HashMap<String, Tuple2<Double, Integer>> col2Statistic = new HashMap<>();

        @Override
        public void endInput() {

            for (Map.Entry<String, Tuple2<Double, Integer>> entry : col2Statistic.entrySet()) {
                String colName = entry.getKey();
                Tuple2<Double, Integer> statisticAndCof = entry.getValue();
                Double statistic = statisticAndCof.f0;
                Integer dof = statisticAndCof.f1;
                double pValue = 1;
                if (dof == 0) {
                    statistic = 0D;
                } else {
                    pValue = 1.0 - new ChiSquaredDistribution(dof).cumulativeProbability(statistic);
                }

                double pValueScaled =
                        new BigDecimal(pValue).setScale(11, RoundingMode.HALF_UP).doubleValue();
                double statisticScaled =
                        new BigDecimal(statistic).setScale(11, RoundingMode.HALF_UP).doubleValue();

                output.collect(
                        new StreamRecord<>(Row.of(colName, pValueScaled, statisticScaled, dof)));
            }
        }

        @Override
        public void processElement(StreamRecord<Tuple3<String, Double, Integer>> element) {
            Tuple3<String, Double, Integer> colAndStatisticAndDof = element.getValue();
            String colName = colAndStatisticAndDof.f0;
            Double partialStatistic = colAndStatisticAndDof.f1;
            Integer dof = colAndStatisticAndDof.f2;

            col2Statistic.merge(
                    colName,
                    new Tuple2<>(partialStatistic, dof),
                    (thisOne, otherOne) -> {
                        thisOne.f0 += otherOne.f0;
                        return thisOne;
                    });
        }
    }
}
