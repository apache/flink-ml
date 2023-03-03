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

package org.apache.flink.ml.feature.univariatefeatureselector;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.stats.anovatest.ANOVATest;
import org.apache.flink.ml.stats.chisqtest.ChiSqTest;
import org.apache.flink.ml.stats.fvaluetest.FValueTest;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * An Estimator which selects features based on univariate statistical tests against labels.
 *
 * <p>Currently, Flink supports three Univariate Feature Selectors: chi-squared, ANOVA F-test and
 * F-value. User can choose Univariate Feature Selector by setting `featureType` and `labelType`,
 * and Flink will pick the score function based on the specified `featureType` and `labelType`.
 *
 * <p>The following combination of `featureType` and `labelType` are supported:
 *
 * <ul>
 *   <li>`featureType` `categorical` and `labelType` `categorical`: Flink uses chi-squared, i.e.
 *       chi2 in sklearn.
 *   <li>`featureType` `continuous` and `labelType` `categorical`: Flink uses ANOVA F-test, i.e.
 *       f_classif in sklearn.
 *   <li>`featureType` `continuous` and `labelType` `continuous`: Flink uses F-value, i.e.
 *       f_regression in sklearn.
 * </ul>
 *
 * <p>The `UnivariateFeatureSelector` supports different selection modes:
 *
 * <ul>
 *   <li>numTopFeatures: chooses a fixed number of top features according to a hypothesis.
 *   <li>percentile: similar to numTopFeatures but chooses a fraction of all features instead of a
 *       fixed number.
 *   <li>fpr: chooses all features whose p-value are below a threshold, thus controlling the false
 *       positive rate of selection.
 *   <li>fdr: uses the <a
 *       href="https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure">
 *       Benjamini-Hochberg procedure</a> to choose all features whose false discovery rate is below
 *       a threshold.
 *   <li>fwe: chooses all features whose p-values are below a threshold. The threshold is scaled by
 *       1/numFeatures, thus controlling the family-wise error rate of selection.
 * </ul>
 *
 * <p>By default, the selection mode is `numTopFeatures`.
 */
public class UnivariateFeatureSelector
        implements Estimator<UnivariateFeatureSelector, UnivariateFeatureSelectorModel>,
                UnivariateFeatureSelectorParams<UnivariateFeatureSelector> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public UnivariateFeatureSelector() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public UnivariateFeatureSelectorModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        final String featuresCol = getFeaturesCol();
        final String labelCol = getLabelCol();
        final String featureType = getFeatureType();
        final String labelType = getLabelType();

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        Table output;
        if (CATEGORICAL.equals(featureType) && CATEGORICAL.equals(labelType)) {
            output =
                    new ChiSqTest()
                            .setFeaturesCol(featuresCol)
                            .setLabelCol(labelCol)
                            .setFlatten(true)
                            .transform(inputs[0])[0];
        } else if (CONTINUOUS.equals(featureType) && CATEGORICAL.equals(labelType)) {
            output =
                    new ANOVATest()
                            .setFeaturesCol(featuresCol)
                            .setLabelCol(labelCol)
                            .setFlatten(true)
                            .transform(inputs[0])[0];
        } else if (CONTINUOUS.equals(featureType) && CONTINUOUS.equals(labelType)) {
            output =
                    new FValueTest()
                            .setFeaturesCol(featuresCol)
                            .setLabelCol(labelCol)
                            .setFlatten(true)
                            .transform(inputs[0])[0];
        } else {
            throw new IllegalArgumentException(
                    String.format(
                            "Unsupported combination: featureType=%s, labelType=%s.",
                            featureType, labelType));
        }
        DataStream<UnivariateFeatureSelectorModelData> modelData =
                tEnv.toDataStream(output)
                        .transform(
                                "selectIndicesFromPValues",
                                TypeInformation.of(UnivariateFeatureSelectorModelData.class),
                                new SelectIndicesFromPValuesOperator(
                                        getSelectionMode(), getActualSelectionThreshold()))
                        .setParallelism(1);
        UnivariateFeatureSelectorModel model =
                new UnivariateFeatureSelectorModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    private double getActualSelectionThreshold() {
        Double threshold = getSelectionThreshold();
        if (threshold == null) {
            String selectionMode = getSelectionMode();
            if (NUM_TOP_FEATURES.equals(selectionMode)) {
                threshold = 50.0;
            } else if (PERCENTILE.equals(selectionMode)) {
                threshold = 0.1;
            } else {
                threshold = 0.05;
            }
        } else {
            if (NUM_TOP_FEATURES.equals(getSelectionMode())) {
                Preconditions.checkArgument(
                        threshold >= 1 && threshold.intValue() == threshold,
                        "SelectionThreshold needs to be a positive Integer "
                                + "for selection mode numTopFeatures, but got %s.",
                        threshold);
            } else {
                Preconditions.checkArgument(
                        threshold >= 0 && threshold <= 1,
                        "SelectionThreshold needs to be in the range [0, 1] "
                                + "for selection mode %s, but got %s.",
                        getSelectionMode(),
                        threshold);
            }
        }
        return threshold;
    }

    private static class SelectIndicesFromPValuesOperator
            extends AbstractStreamOperator<UnivariateFeatureSelectorModelData>
            implements OneInputStreamOperator<Row, UnivariateFeatureSelectorModelData>,
                    BoundedOneInput {
        private final String selectionMode;
        private final double threshold;

        private List<Tuple2<Double, Integer>> pValuesAndIndices;
        private ListState<Tuple2<Double, Integer>> pValuesAndIndicesState;

        public SelectIndicesFromPValuesOperator(String selectionMode, double threshold) {
            this.selectionMode = selectionMode;
            this.threshold = threshold;
        }

        @Override
        public void endInput() {
            List<Integer> indices = new ArrayList<>();

            switch (selectionMode) {
                case NUM_TOP_FEATURES:
                    pValuesAndIndices.sort(
                            Comparator.comparingDouble((Tuple2<Double, Integer> t) -> t.f0)
                                    .thenComparingInt(t -> t.f1));
                    IntStream.range(0, Math.min(pValuesAndIndices.size(), (int) threshold))
                            .forEach(i -> indices.add(pValuesAndIndices.get(i).f1));
                    break;
                case PERCENTILE:
                    pValuesAndIndices.sort(
                            Comparator.comparingDouble((Tuple2<Double, Integer> t) -> t.f0)
                                    .thenComparingInt(t -> t.f1));
                    IntStream.range(
                                    0,
                                    Math.min(
                                            pValuesAndIndices.size(),
                                            (int) (pValuesAndIndices.size() * threshold)))
                            .forEach(i -> indices.add(pValuesAndIndices.get(i).f1));
                    break;
                case FPR:
                    pValuesAndIndices.stream()
                            .filter(x -> x.f0 < threshold)
                            .forEach(x -> indices.add(x.f1));
                    break;
                case FDR:
                    pValuesAndIndices.sort(
                            Comparator.comparingDouble((Tuple2<Double, Integer> t) -> t.f0)
                                    .thenComparingInt(t -> t.f1));

                    int maxIndex = -1;
                    for (int i = 0; i < pValuesAndIndices.size(); i++) {
                        if (pValuesAndIndices.get(i).f0
                                < (threshold / pValuesAndIndices.size()) * (i + 1)) {
                            maxIndex = Math.max(maxIndex, i);
                        }
                    }
                    if (maxIndex >= 0) {
                        pValuesAndIndices.sort(
                                Comparator.comparingDouble((Tuple2<Double, Integer> t) -> t.f0)
                                        .thenComparingInt(t -> t.f1));
                        IntStream.range(0, maxIndex + 1)
                                .forEach(i -> indices.add(pValuesAndIndices.get(i).f1));
                    }
                    break;
                case FWE:
                    pValuesAndIndices.stream()
                            .filter(x -> x.f0 < threshold / pValuesAndIndices.size())
                            .forEach(x -> indices.add(x.f1));
                    break;
                default:
                    throw new RuntimeException("Unknown Selection Mode: " + selectionMode);
            }

            UnivariateFeatureSelectorModelData modelData =
                    new UnivariateFeatureSelectorModelData(
                            indices.stream().mapToInt(Integer::intValue).toArray());
            output.collect(new StreamRecord<>(modelData));
        }

        @Override
        public void processElement(StreamRecord<Row> record) {
            Row row = record.getValue();
            double pValue = (double) row.getField("pValue");
            int featureIndex = (int) row.getField("featureIndex");
            pValuesAndIndices.add(Tuple2.of(pValue, featureIndex));
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            pValuesAndIndicesState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "pValuesAndIndices",
                                            Types.TUPLE(Types.DOUBLE, Types.INT)));
            pValuesAndIndices = IteratorUtils.toList(pValuesAndIndicesState.get().iterator());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            pValuesAndIndicesState.update(pValuesAndIndices);
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

    public static UnivariateFeatureSelector load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
