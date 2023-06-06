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

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.param.HasFlatten;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
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
 * An AlgoOperator which implements the Chi-square test algorithm.
 *
 * <p>Chi-square Test computes the statistics of independence of variables in a contingency table,
 * e.g., p-value, and DOF(degree of freedom) for each input feature. The contingency table is
 * constructed from the observed categorical values.
 *
 * <p>The input of this algorithm is a table containing a labelColumn of numerical type and a
 * featuresColumn of vector type. Each index in the input vector represents a feature to be tested.
 * By default, the output of this algorithm is a table containing a single row with the following
 * columns, each of which has one value per feature.
 *
 * <ul>
 *   <li>"pValues": vector
 *   <li>"degreesOfFreedom": int array
 *   <li>"statistics": vector
 * </ul>
 *
 * <p>The output of this algorithm can be flattened to multiple rows by setting {@link
 * HasFlatten#FLATTEN} to true, which would contain the following columns:
 *
 * <ul>
 *   <li>"featureIndex": int
 *   <li>"pValue": double
 *   <li>"degreeOfFreedom": int
 *   <li>"statistic": double
 * </ul>
 *
 * <p>See: http://en.wikipedia.org/wiki/Chi-squared_test.
 */
public class ChiSqTest implements AlgoOperator<ChiSqTest>, ChiSqTestParams<ChiSqTest> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public ChiSqTest() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        final String bcCategoricalMarginsKey = "bcCategoricalMarginsKey";
        final String bcLabelMarginsKey = "bcLabelMarginsKey";

        final String featuresCol = getFeaturesCol();
        final String labelCol = getLabelCol();

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        SingleOutputStreamOperator<Tuple3<Integer, Double, Double>> indexAndFeatureAndLabel =
                tEnv.toDataStream(inputs[0])
                        .flatMap(new ExtractIndexAndFeatureAndLabel(featuresCol, labelCol));

        DataStream<Tuple4<Integer, Double, Double, Long>> observedFreq =
                indexAndFeatureAndLabel
                        .keyBy(Tuple3::hashCode)
                        .transform(
                                "GenerateObservedFrequencies",
                                Types.TUPLE(Types.INT, Types.DOUBLE, Types.DOUBLE, Types.LONG),
                                new GenerateObservedFrequencies());

        SingleOutputStreamOperator<Tuple4<Integer, Double, Double, Long>> filledObservedFreq =
                observedFreq
                        .transform(
                                "filledObservedFreq",
                                Types.TUPLE(Types.INT, Types.DOUBLE, Types.DOUBLE, Types.LONG),
                                new FillFrequencyTable())
                        .setParallelism(1);

        DataStream<Tuple3<Integer, Double, Long>> categoricalMargins =
                observedFreq
                        .keyBy(tuple -> new Tuple2<>(tuple.f0, tuple.f1).hashCode())
                        .transform(
                                "AggregateCategoricalMargins",
                                Types.TUPLE(Types.INT, Types.DOUBLE, Types.LONG),
                                new AggregateCategoricalMargins());

        DataStream<Tuple3<Integer, Double, Long>> labelMargins =
                observedFreq
                        .keyBy(tuple -> new Tuple2<>(tuple.f0, tuple.f2).hashCode())
                        .transform(
                                "AggregateLabelMargins",
                                Types.TUPLE(Types.INT, Types.DOUBLE, Types.LONG),
                                new AggregateLabelMargins());

        Function<List<DataStream<?>>, DataStream<Tuple3<Integer, Double, Integer>>> function =
                dataStreams -> {
                    DataStream stream = dataStreams.get(0);
                    return stream.map(
                            new ChiSqFunc(bcCategoricalMarginsKey, bcLabelMarginsKey),
                            Types.TUPLE(Types.INT, Types.DOUBLE, Types.INT));
                };

        HashMap<String, DataStream<?>> bcMap =
                new HashMap<String, DataStream<?>>() {
                    {
                        put(bcCategoricalMarginsKey, categoricalMargins);
                        put(bcLabelMarginsKey, labelMargins);
                    }
                };

        DataStream<Tuple3<Integer, Double, Integer>> categoricalStatistics =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(filledObservedFreq), bcMap, function);

        boolean flatten = getFlatten();

        RowTypeInfo outputTypeInfo;
        if (flatten) {
            outputTypeInfo =
                    new RowTypeInfo(
                            new TypeInformation[] {
                                Types.INT, Types.DOUBLE, Types.INT, Types.DOUBLE
                            },
                            new String[] {
                                "featureIndex", "pValue", "degreeOfFreedom", "statistic"
                            });
        } else {
            outputTypeInfo =
                    new RowTypeInfo(
                            new TypeInformation[] {
                                DenseIntDoubleVectorTypeInfo.INSTANCE,
                                Types.PRIMITIVE_ARRAY(Types.INT),
                                DenseIntDoubleVectorTypeInfo.INSTANCE
                            },
                            new String[] {"pValues", "degreesOfFreedom", "statistics"});
        }

        SingleOutputStreamOperator<Row> chiSqTestResult =
                categoricalStatistics
                        .transform(
                                "chiSqTestResult", outputTypeInfo, new AggregateChiSqFunc(flatten))
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

    private static class ExtractIndexAndFeatureAndLabel
            extends RichFlatMapFunction<Row, Tuple3<Integer, Double, Double>> {
        private final String featuresCol;
        private final String labelCol;

        public ExtractIndexAndFeatureAndLabel(String featuresCol, String labelCol) {
            this.featuresCol = featuresCol;
            this.labelCol = labelCol;
        }

        @Override
        public void flatMap(Row row, Collector<Tuple3<Integer, Double, Double>> collector) {

            Double label = ((Number) row.getFieldAs(labelCol)).doubleValue();

            IntDoubleVector features = row.getFieldAs(featuresCol);
            for (int i = 0; i < features.size(); i++) {
                collector.collect(Tuple3.of(i, features.get(i), label));
            }
        }
    }

    /**
     * Computes the frequency of each feature value at different indices by labels. An output record
     * (indexA, featureValueB, labelC, countD) represents that A feature value {featureValueB} with
     * label {labelC} at index {indexA} has appeared {countD} times in the input table.
     */
    private static class GenerateObservedFrequencies
            extends AbstractStreamOperator<Tuple4<Integer, Double, Double, Long>>
            implements OneInputStreamOperator<
                            Tuple3<Integer, Double, Double>, Tuple4<Integer, Double, Double, Long>>,
                    BoundedOneInput {

        private Map<Tuple3<Integer, Double, Double>, Long> cntMap = new HashMap<>();
        private ListState<Map<Tuple3<Integer, Double, Double>, Long>> cntMapState;

        @Override
        public void endInput() {
            for (Tuple3<Integer, Double, Double> key : cntMap.keySet()) {
                Long count = cntMap.get(key);
                output.collect(new StreamRecord<>(new Tuple4<>(key.f0, key.f1, key.f2, count)));
            }
            cntMapState.clear();
        }

        @Override
        public void processElement(StreamRecord<Tuple3<Integer, Double, Double>> element) {

            Tuple3<Integer, Double, Double> indexAndCategoryAndLabel = element.getValue();
            cntMap.compute(indexAndCategoryAndLabel, (k, v) -> (v == null ? 1 : v + 1));
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            cntMapState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "cntMapState",
                                            Types.MAP(
                                                    Types.TUPLE(
                                                            Types.INT, Types.DOUBLE, Types.DOUBLE),
                                                    Types.LONG)));

            OperatorStateUtils.getUniqueElement(cntMapState, "cntMapState")
                    .ifPresent(x -> cntMap = x);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            cntMapState.update(Collections.singletonList(cntMap));
        }
    }

    /**
     * Fills the frequency table by setting the frequency of missed elements (i.e., missed
     * combinations of index, featureValue and labelValue) as zero.
     */
    private static class FillFrequencyTable
            extends AbstractStreamOperator<Tuple4<Integer, Double, Double, Long>>
            implements OneInputStreamOperator<
                            Tuple4<Integer, Double, Double, Long>,
                            Tuple4<Integer, Double, Double, Long>>,
                    BoundedOneInput {

        private Map<Tuple2<Integer, Double>, List<Tuple2<Double, Long>>> valuesMap =
                new HashMap<>();
        private HashSet<Double> distinctLabels = new HashSet<>();

        private ListState<Map<Tuple2<Integer, Double>, List<Tuple2<Double, Long>>>> valuesMapState;
        private ListState<List<Double>> distinctLabelsState;

        @Override
        public void endInput() {

            for (Map.Entry<Tuple2<Integer, Double>, List<Tuple2<Double, Long>>> entry :
                    valuesMap.entrySet()) {
                List<Tuple2<Double, Long>> labelAndCountList = entry.getValue();
                Tuple2<Integer, Double> categoricalKey = entry.getKey();

                List<Double> existingLabels =
                        labelAndCountList.stream().map(v -> v.f0).collect(Collectors.toList());

                for (Double label : distinctLabels) {
                    if (!existingLabels.contains(label)) {
                        Tuple2<Double, Long> generatedLabelCount = new Tuple2<>(label, 0L);
                        labelAndCountList.add(generatedLabelCount);
                    }
                }

                for (Tuple2<Double, Long> labelAndCount : labelAndCountList) {
                    output.collect(
                            new StreamRecord<>(
                                    new Tuple4<>(
                                            categoricalKey.f0,
                                            categoricalKey.f1,
                                            labelAndCount.f0,
                                            labelAndCount.f1)));
                }
            }

            valuesMapState.clear();
            distinctLabelsState.clear();
        }

        @Override
        public void processElement(StreamRecord<Tuple4<Integer, Double, Double, Long>> element) {
            Tuple4<Integer, Double, Double, Long> indexAndCategoryAndLabelAndCount =
                    element.getValue();
            Tuple2<Integer, Double> key =
                    new Tuple2<>(
                            indexAndCategoryAndLabelAndCount.f0,
                            indexAndCategoryAndLabelAndCount.f1);
            Tuple2<Double, Long> labelAndCount =
                    new Tuple2<>(
                            indexAndCategoryAndLabelAndCount.f2,
                            indexAndCategoryAndLabelAndCount.f3);
            List<Tuple2<Double, Long>> labelAndCountList = valuesMap.get(key);

            if (labelAndCountList == null) {
                ArrayList<Tuple2<Double, Long>> value = new ArrayList<>();
                value.add(labelAndCount);
                valuesMap.put(key, value);
            } else {
                labelAndCountList.add(labelAndCount);
            }

            distinctLabels.add(indexAndCategoryAndLabelAndCount.f2);
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            valuesMapState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "valuesMapState",
                                            Types.MAP(
                                                    Types.TUPLE(Types.INT, Types.DOUBLE),
                                                    Types.LIST(
                                                            Types.TUPLE(
                                                                    Types.DOUBLE, Types.LONG)))));
            distinctLabelsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "distinctLabelsState", Types.LIST(Types.DOUBLE)));

            OperatorStateUtils.getUniqueElement(valuesMapState, "valuesMapState")
                    .ifPresent(x -> valuesMap = x);

            OperatorStateUtils.getUniqueElement(distinctLabelsState, "distinctLabelsState")
                    .ifPresent(x -> distinctLabels = new HashSet<>(x));
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            valuesMapState.update(Collections.singletonList(valuesMap));
            distinctLabelsState.update(Collections.singletonList(new ArrayList<>(distinctLabels)));
        }
    }

    /** Computes the marginal sums of different categories. */
    private static class AggregateCategoricalMargins
            extends AbstractStreamOperator<Tuple3<Integer, Double, Long>>
            implements OneInputStreamOperator<
                            Tuple4<Integer, Double, Double, Long>, Tuple3<Integer, Double, Long>>,
                    BoundedOneInput {

        private Map<Tuple2<Integer, Double>, Long> categoricalMarginsMap = new HashMap<>();

        private ListState<Map<Tuple2<Integer, Double>, Long>> categoricalMarginsMapState;

        @Override
        public void endInput() {
            for (Tuple2<Integer, Double> key : categoricalMarginsMap.keySet()) {
                Long categoricalMargin = categoricalMarginsMap.get(key);
                output.collect(new StreamRecord<>(new Tuple3<>(key.f0, key.f1, categoricalMargin)));
            }
            categoricalMarginsMap.clear();
        }

        @Override
        public void processElement(StreamRecord<Tuple4<Integer, Double, Double, Long>> element) {

            Tuple4<Integer, Double, Double, Long> indexAndCategoryAndLabelAndCnt =
                    element.getValue();
            Tuple2<Integer, Double> key =
                    new Tuple2<>(
                            indexAndCategoryAndLabelAndCnt.f0, indexAndCategoryAndLabelAndCnt.f1);
            Long observedFreq = indexAndCategoryAndLabelAndCnt.f3;
            categoricalMarginsMap.compute(
                    key, (k, v) -> (v == null ? observedFreq : v + observedFreq));
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            categoricalMarginsMapState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "categoricalMarginsMapState",
                                            Types.MAP(
                                                    Types.TUPLE(Types.INT, Types.DOUBLE),
                                                    Types.LONG)));

            OperatorStateUtils.getUniqueElement(
                            categoricalMarginsMapState, "categoricalMarginsMapState")
                    .ifPresent(x -> categoricalMarginsMap = x);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            categoricalMarginsMapState.update(Collections.singletonList(categoricalMarginsMap));
        }
    }

    /** Computes the marginal sums of different labels. */
    private static class AggregateLabelMargins
            extends AbstractStreamOperator<Tuple3<Integer, Double, Long>>
            implements OneInputStreamOperator<
                            Tuple4<Integer, Double, Double, Long>, Tuple3<Integer, Double, Long>>,
                    BoundedOneInput {

        private Map<Tuple2<Integer, Double>, Long> labelMarginsMap = new HashMap<>();
        private ListState<Map<Tuple2<Integer, Double>, Long>> labelMarginsMapState;

        @Override
        public void endInput() {

            for (Tuple2<Integer, Double> key : labelMarginsMap.keySet()) {
                Long labelMargin = labelMarginsMap.get(key);
                output.collect(new StreamRecord<>(new Tuple3<>(key.f0, key.f1, labelMargin)));
            }
            labelMarginsMapState.clear();
        }

        @Override
        public void processElement(StreamRecord<Tuple4<Integer, Double, Double, Long>> element) {

            Tuple4<Integer, Double, Double, Long> indexAndFeatureAndLabelAndCnt =
                    element.getValue();
            Long observedFreq = indexAndFeatureAndLabelAndCnt.f3;
            Tuple2<Integer, Double> key =
                    new Tuple2<>(
                            indexAndFeatureAndLabelAndCnt.f0, indexAndFeatureAndLabelAndCnt.f2);

            labelMarginsMap.compute(key, (k, v) -> (v == null ? observedFreq : v + observedFreq));
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            labelMarginsMapState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "labelMarginsMapState",
                                            Types.MAP(
                                                    Types.TUPLE(Types.INT, Types.DOUBLE),
                                                    Types.LONG)));

            OperatorStateUtils.getUniqueElement(labelMarginsMapState, "labelMarginsMapState")
                    .ifPresent(x -> labelMarginsMap = x);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            labelMarginsMapState.update(Collections.singletonList(labelMarginsMap));
        }
    }

    /** Conduct Pearson's independence test on the input contingency table. */
    private static class ChiSqFunc
            extends RichMapFunction<
                    Tuple4<Integer, Double, Double, Long>, Tuple3<Integer, Double, Integer>> {

        private final String bcCategoricalMarginsKey;
        private final String bcLabelMarginsKey;
        private final Map<Tuple2<Integer, Double>, Long> categoricalMargins = new HashMap<>();
        private final Map<Tuple2<Integer, Double>, Long> labelMargins = new HashMap<>();

        double sampleSize = 0;
        int numLabels = 0;
        HashMap<Integer, Integer> index2NumCategories = new HashMap<>();

        public ChiSqFunc(String bcCategoricalMarginsKey, String bcLabelMarginsKey) {
            this.bcCategoricalMarginsKey = bcCategoricalMarginsKey;
            this.bcLabelMarginsKey = bcLabelMarginsKey;
        }

        @Override
        public Tuple3<Integer, Double, Integer> map(Tuple4<Integer, Double, Double, Long> v) {
            if (categoricalMargins.isEmpty()) {
                List<Tuple3<Integer, Double, Long>> categoricalMarginList =
                        getRuntimeContext().getBroadcastVariable(bcCategoricalMarginsKey);
                List<Tuple3<Integer, Double, Long>> labelMarginList =
                        getRuntimeContext().getBroadcastVariable(bcLabelMarginsKey);

                for (Tuple3<Integer, Double, Long> indexAndFeatureAndCount :
                        categoricalMarginList) {
                    index2NumCategories.merge(indexAndFeatureAndCount.f0, 1, Integer::sum);
                }

                numLabels = (int) labelMarginList.stream().map(x -> x.f1).distinct().count();

                for (Tuple3<Integer, Double, Long> indexAndFeatureAndCount :
                        categoricalMarginList) {
                    categoricalMargins.put(
                            new Tuple2<>(indexAndFeatureAndCount.f0, indexAndFeatureAndCount.f1),
                            indexAndFeatureAndCount.f2);
                }

                Map<Integer, Double> sampleSizeCount = new HashMap<>();
                Integer tmpKey = null;

                for (Tuple3<Integer, Double, Long> indexAndLabelAndCount : labelMarginList) {
                    Integer index = indexAndLabelAndCount.f0;

                    if (tmpKey == null) {
                        tmpKey = index;
                        sampleSizeCount.put(index, 0D);
                    }

                    sampleSizeCount.computeIfPresent(
                            index, (k, count) -> count + indexAndLabelAndCount.f2);
                    labelMargins.put(
                            new Tuple2<>(index, indexAndLabelAndCount.f1),
                            indexAndLabelAndCount.f2);
                }

                Optional<Double> sampleSizeOpt =
                        sampleSizeCount.values().stream().reduce(Double::sum);
                Preconditions.checkArgument(sampleSizeOpt.isPresent());
                sampleSize = sampleSizeOpt.get();
            }

            Integer index = v.f0;
            // Degrees of freedom
            int dof = (index2NumCategories.get(index) - 1) * (numLabels - 1);

            Tuple2<Integer, Double> category = new Tuple2<>(v.f0, v.f1);

            Tuple2<Integer, Double> indexAndLabelKey = new Tuple2<>(v.f0, v.f2);
            Long theCategoricalMargin = categoricalMargins.get(category);
            Long theLabelMargin = labelMargins.get(indexAndLabelKey);
            Long observed = v.f3;

            double expected = (double) (theLabelMargin * theCategoricalMargin) / sampleSize;
            double categoricalStatistic = pearsonFunc(observed, expected);

            return new Tuple3<>(index, categoricalStatistic, dof);
        }

        // Pearson's chi-squared test: http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
        private double pearsonFunc(double observed, double expected) {
            double dev = observed - expected;
            return dev * dev / expected;
        }
    }

    /**
     * Computes the Pearson's chi-squared statistic, p-value and the number of degrees of freedom
     * for every feature across the input DataStream.
     */
    private static class AggregateChiSqFunc extends AbstractStreamOperator<Row>
            implements OneInputStreamOperator<Tuple3<Integer, Double, Integer>, Row>,
                    BoundedOneInput {
        private final boolean flatten;
        private Map<Integer, Tuple2<Double, Integer>> index2Statistic = new HashMap<>();
        private ListState<Map<Integer, Tuple2<Double, Integer>>> index2StatisticState;

        private AggregateChiSqFunc(boolean flatten) {
            this.flatten = flatten;
        }

        @Override
        public void endInput() {
            if (flatten) {
                endInputWithFlatten();
            } else {
                endInputWithoutFlatten();
            }
        }

        private void endInputWithFlatten() {
            for (Map.Entry<Integer, Tuple2<Double, Integer>> entry : index2Statistic.entrySet()) {
                int index = entry.getKey();
                Tuple3<Double, Integer, Double> pValueAndDofAndStatistic =
                        computePValueAndScale(entry.getValue());
                output.collect(
                        new StreamRecord<>(
                                Row.of(
                                        index,
                                        pValueAndDofAndStatistic.f0,
                                        pValueAndDofAndStatistic.f1,
                                        pValueAndDofAndStatistic.f2)));
            }
        }

        private void endInputWithoutFlatten() {
            int size = index2Statistic.size();
            IntDoubleVector pValueScaledVector = new DenseIntDoubleVector(size);
            IntDoubleVector statisticScaledVector = new DenseIntDoubleVector(size);
            int[] dofArray = new int[size];

            for (Map.Entry<Integer, Tuple2<Double, Integer>> entry : index2Statistic.entrySet()) {
                int index = entry.getKey();
                Tuple3<Double, Integer, Double> pValueAndDofAndStatistic =
                        computePValueAndScale(entry.getValue());
                pValueScaledVector.set(index, pValueAndDofAndStatistic.f0);
                statisticScaledVector.set(index, pValueAndDofAndStatistic.f2);
                dofArray[index] = pValueAndDofAndStatistic.f1;
            }

            output.collect(
                    new StreamRecord<>(
                            Row.of(pValueScaledVector, dofArray, statisticScaledVector)));
        }

        private static Tuple3<Double, Integer, Double> computePValueAndScale(
                Tuple2<Double, Integer> statisticAndDof) {
            Double statistic = statisticAndDof.f0;
            Integer dof = statisticAndDof.f1;
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
            return Tuple3.of(pValueScaled, dof, statisticScaled);
        }

        @Override
        public void processElement(StreamRecord<Tuple3<Integer, Double, Integer>> element) {
            Tuple3<Integer, Double, Integer> indexAndStatisticAndDof = element.getValue();
            Integer index = indexAndStatisticAndDof.f0;
            Double partialStatistic = indexAndStatisticAndDof.f1;
            Integer dof = indexAndStatisticAndDof.f2;

            index2Statistic.merge(
                    index,
                    new Tuple2<>(partialStatistic, dof),
                    (thisOne, otherOne) -> {
                        thisOne.f0 += otherOne.f0;
                        return thisOne;
                    });
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            index2StatisticState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "index2StatisticState",
                                            Types.MAP(
                                                    Types.INT,
                                                    Types.TUPLE(Types.DOUBLE, Types.INT))));

            OperatorStateUtils.getUniqueElement(index2StatisticState, "index2StatisticState")
                    .ifPresent(x -> index2Statistic = x);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            index2StatisticState.update(Collections.singletonList(index2Statistic));
        }
    }
}
