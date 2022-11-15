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

package org.apache.flink.ml.feature.chisqselector;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.stats.chisqtest.ChiSqTest;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.catalog.DataTypeFactory;
import org.apache.flink.table.functions.ScalarFunction;
import org.apache.flink.table.types.inference.TypeInference;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeSet;
import java.util.stream.Collectors;

import static org.apache.flink.table.api.Expressions.$;
import static org.apache.flink.table.api.Expressions.call;

/**
 * ChiSqSelector is an algorithm that selects categorical features to use for predicting a
 * categorical label.
 *
 * <p>The selector supports different selection methods as follows.
 *
 * <ul>
 *   <li>`numTopFeatures` chooses a fixed number of top features according to a chi-squared test.
 *   <li>`percentile` is similar but chooses a fraction of all features instead of a fixed number.
 *   <li>`fpr` chooses all features whose p-value are below a threshold, thus controlling the false
 *       positive rate of selection.
 *   <li>`fdr` uses the [Benjamini-Hochberg procedure]
 *       (https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure)
 *       to choose all features whose false discovery rate is below a threshold.
 *   <li>`fwe` chooses all features whose p-values are below a threshold. The threshold is scaled by
 *       1/numFeatures, thus controlling the family-wise error rate of selection.
 * </ul>
 *
 * <p>By default, the selection method is `numTopFeatures`, with the default number of top features
 * set to 50.
 */
public class ChiSqSelector
        implements Estimator<ChiSqSelector, ChiSqSelectorModel>,
                ChiSqSelectorParams<ChiSqSelector> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public ChiSqSelector() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public ChiSqSelectorModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        ChiSqTest chiSqTest =
                new ChiSqTest()
                        .setFeaturesCol(getFeaturesCol())
                        .setLabelCol(getLabelCol())
                        .setFlatten(false);

        GenerateModelDataFunction function =
                new GenerateModelDataFunction(
                        getSelectorType(),
                        getNumTopFeatures(),
                        getPercentile(),
                        getFpr(),
                        getFdr(),
                        getFwe());

        Table modelDataTable =
                chiSqTest.transform(inputs)[0].select(
                        call(function, $("pValues")).as("selectedFeatures"));

        ChiSqSelectorModel model = new ChiSqSelectorModel().setModelData(modelDataTable);
        ReadWriteUtils.updateExistingParams(model, getParamMap());

        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static ChiSqSelector load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * A scalar function that computes the model data for {@link ChiSqSelectorModel} according to
     * the statistics computed by Chi-square test algorithm.
     */
    public static class GenerateModelDataFunction extends ScalarFunction {
        private final String selectorType;
        private final int numTopFeatures;
        private final double percentile;
        private final double fpr;
        private final double fdr;
        private final double fwe;

        private GenerateModelDataFunction(
                String selectorType,
                int numTopFeatures,
                double percentile,
                double fpr,
                double fdr,
                double fwe) {
            this.selectorType = selectorType;
            this.numTopFeatures = numTopFeatures;
            this.percentile = percentile;
            this.fpr = fpr;
            this.fdr = fdr;
            this.fwe = fwe;
        }

        public int[] eval(Vector pValues) {
            List<Integer> selectedFeatures;
            switch (selectorType) {
                case NUM_TOP_FEATURES_TYPE:
                    selectedFeatures = getTopIndices(pValues, numTopFeatures);
                    break;
                case PERCENTILE_TYPE:
                    selectedFeatures = getTopIndices(pValues, (int) (pValues.size() * percentile));
                    break;
                case FPR_TYPE:
                    selectedFeatures = getIndicesByThreshold(pValues, fpr);
                    break;
                case FDR_TYPE:
                    // This uses the Benjamini-Hochberg procedure.
                    // https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure
                    double f = fdr / pValues.size();
                    double[] sortedPValues = pValues.toArray().clone();
                    Arrays.sort(sortedPValues);
                    int maxIndex = -1;
                    for (int i = 0; i < sortedPValues.length; i++) {
                        if (sortedPValues[i] <= f * (i + 1)) {
                            maxIndex = i;
                        }
                    }
                    if (maxIndex >= 0) {
                        selectedFeatures = getTopIndices(pValues, maxIndex + 1);
                    } else {
                        selectedFeatures = Collections.EMPTY_LIST;
                    }
                    break;
                case FWE_TYPE:
                    selectedFeatures = getIndicesByThreshold(pValues, fwe / pValues.size());
                    break;
                default:
                    throw new UnsupportedOperationException(
                            String.format("Unsupported selector type %s.", selectorType));
            }
            return selectedFeatures.stream().sorted().mapToInt(x -> x).toArray();
        }

        @Override
        public TypeInference getTypeInference(DataTypeFactory typeFactory) {
            return TypeInference.newBuilder()
                    .outputTypeStrategy(
                            callContext ->
                                    Optional.of(
                                            DataTypes.ARRAY(DataTypes.INT().notNull())
                                                    .bridgedTo(int[].class)))
                    .build();
        }
    }

    private static List<Integer> getTopIndices(Vector pValues, int k) {
        TreeSet<Tuple2<Integer, Double>> set = new TreeSet<>(new TopFeatureComparator());
        for (int i = 0; i < pValues.size(); i++) {
            set.add(Tuple2.of(i, pValues.get(i)));
            if (set.size() > k) {
                set.pollLast();
            }
        }

        return set.stream().map(x -> x.f0).collect(Collectors.toList());
    }

    private static List<Integer> getIndicesByThreshold(Vector pValues, double threshold) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < pValues.size(); i++) {
            if (pValues.get(i) < threshold) {
                indices.add(i);
            }
        }
        return indices;
    }

    private static class TopFeatureComparator implements Comparator<Tuple2<Integer, Double>> {
        @Override
        public int compare(
                Tuple2<Integer, Double> indexAndPValue1, Tuple2<Integer, Double> indexAndPValue2) {
            if (indexAndPValue1.f1 > indexAndPValue2.f1) {
                return 1;
            } else if (indexAndPValue1.f1 < indexAndPValue2.f1) {
                return -1;
            } else {
                return indexAndPValue1.f0 - indexAndPValue2.f0;
            }
        }
    }
}
