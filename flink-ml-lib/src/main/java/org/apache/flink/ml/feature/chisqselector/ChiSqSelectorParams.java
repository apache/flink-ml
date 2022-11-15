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

import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link ChiSqSelector}.
 *
 * @param <T> The class type of this instance.
 */
public interface ChiSqSelectorParams<T> extends ChiSqSelectorModelParams<T>, HasLabelCol<T> {
    String NUM_TOP_FEATURES_TYPE = "numTopFeatures";
    String PERCENTILE_TYPE = "percentile";
    String FPR_TYPE = "fpr";
    String FDR_TYPE = "fdr";
    String FWE_TYPE = "fwe";

    Param<String> SELECTOR_TYPE =
            new StringParam(
                    "selectorType",
                    "The selector type. Supported options: numTopFeatures, percentile, fpr, fdr, fwe.",
                    NUM_TOP_FEATURES_TYPE,
                    ParamValidators.inArray(
                            NUM_TOP_FEATURES_TYPE, PERCENTILE_TYPE, FPR_TYPE, FDR_TYPE, FWE_TYPE));

    Param<Integer> NUM_TOP_FEATURES =
            new IntParam(
                    "numTopFeatures",
                    "Number of features that selector will select, ordered by ascending p-value. If the"
                            + " number of features is < numTopFeatures, then this will select all features.",
                    50,
                    ParamValidators.gtEq(1));

    Param<Double> PERCENTILE =
            new DoubleParam(
                    "percentile",
                    "Percentile of features that selector will select, ordered by ascending p-value.",
                    0.1,
                    ParamValidators.inRange(0, 1));

    Param<Double> FPR =
            new DoubleParam(
                    "fpr",
                    "The highest p-value for features to be kept.",
                    0.05,
                    ParamValidators.inRange(0, 1));

    Param<Double> FDR =
            new DoubleParam(
                    "fdr",
                    "The upper bound of the expected false discovery rate.",
                    0.05,
                    ParamValidators.inRange(0, 1));

    Param<Double> FWE =
            new DoubleParam(
                    "fwe",
                    "The upper bound of the expected family-wise error rate.",
                    0.05,
                    ParamValidators.inRange(0, 1));

    default String getSelectorType() {
        return get(SELECTOR_TYPE);
    }

    default T setSelectorType(String value) {
        return set(SELECTOR_TYPE, value);
    }

    default int getNumTopFeatures() {
        return get(NUM_TOP_FEATURES);
    }

    default T setNumTopFeatures(int value) {
        return set(NUM_TOP_FEATURES, value);
    }

    default double getPercentile() {
        return get(PERCENTILE);
    }

    default T setPercentile(double value) {
        return set(PERCENTILE, value);
    }

    default double getFpr() {
        return get(FPR);
    }

    default T setFpr(double value) {
        return set(FPR, value);
    }

    default double getFdr() {
        return get(FDR);
    }

    default T setFdr(double value) {
        return set(FDR, value);
    }

    default double getFwe() {
        return get(FWE);
    }

    default T setFwe(double value) {
        return set(FWE, value);
    }
}
