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

import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link UnivariateFeatureSelector}.
 *
 * @param <T> The class type of this instance.
 */
public interface UnivariateFeatureSelectorParams<T>
        extends HasLabelCol<T>, UnivariateFeatureSelectorModelParams<T> {

    String CATEGORICAL = "categorical";
    String CONTINUOUS = "continuous";

    String NUM_TOP_FEATURES = "numTopFeatures";
    String PERCENTILE = "percentile";
    String FPR = "fpr";
    String FDR = "fdr";
    String FWE = "fwe";

    /**
     * Supported options of the feature type.
     *
     * <ul>
     *   <li>categorical: the features are categorical data.
     *   <li>continuous: the features are continuous data.
     * </ul>
     */
    Param<String> FEATURE_TYPE =
            new StringParam(
                    "featureType",
                    "The feature type.",
                    null,
                    ParamValidators.inArray(CATEGORICAL, CONTINUOUS));

    /**
     * Supported options of the label type.
     *
     * <ul>
     *   <li>categorical: the label is categorical data.
     *   <li>continuous: the label is continuous data.
     * </ul>
     */
    Param<String> LABEL_TYPE =
            new StringParam(
                    "labelType",
                    "The label type.",
                    null,
                    ParamValidators.inArray(CATEGORICAL, CONTINUOUS));

    /**
     * Supported options of the feature selection mode.
     *
     * <ul>
     *   <li>numTopFeatures: chooses a fixed number of top features according to a hypothesis.
     *   <li>percentile: similar to numTopFeatures but chooses a fraction of all features instead of
     *       a fixed number.
     *   <li>fpr: chooses all features whose p-value are below a threshold, thus controlling the
     *       false positive rate of selection.
     *   <li>fdr: uses the <a
     *       href="https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini.E2.80.93Hochberg_procedure">
     *       Benjamini-Hochberg procedure</a> to choose all features whose false discovery rate is
     *       below a threshold.
     *   <li>fwe: chooses all features whose p-values are below a threshold. The threshold is scaled
     *       by 1/numFeatures, thus controlling the family-wise error rate of selection.
     * </ul>
     */
    Param<String> SELECTION_MODE =
            new StringParam(
                    "selectionMode",
                    "The feature selection mode.",
                    NUM_TOP_FEATURES,
                    ParamValidators.inArray(NUM_TOP_FEATURES, PERCENTILE, FPR, FDR, FWE));

    Param<Double> SELECTION_THRESHOLD =
            new DoubleParam(
                    "selectionThreshold",
                    "The upper bound of the features that selector will select. If not set, "
                            + "it will be replaced with a meaningful value according to different "
                            + "selection modes at runtime. When the mode is numTopFeatures, it will be "
                            + "replaced with 50; when the mode is percentile, it will be replaced "
                            + "with 0.1; otherwise, it will be replaced with 0.05.",
                    null);

    default String getFeatureType() {
        return get(FEATURE_TYPE);
    }

    default T setFeatureType(String value) {
        return set(FEATURE_TYPE, value);
    }

    default String getLabelType() {
        return get(LABEL_TYPE);
    }

    default T setLabelType(String value) {
        return set(LABEL_TYPE, value);
    }

    default String getSelectionMode() {
        return get(SELECTION_MODE);
    }

    default T setSelectionMode(String value) {
        return set(SELECTION_MODE, value);
    }

    default Double getSelectionThreshold() {
        return get(SELECTION_THRESHOLD);
    }

    default T setSelectionThreshold(double value) {
        return set(SELECTION_THRESHOLD, value);
    }
}
