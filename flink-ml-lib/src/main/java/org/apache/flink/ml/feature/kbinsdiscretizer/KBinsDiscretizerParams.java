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

package org.apache.flink.ml.feature.kbinsdiscretizer;

import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link KBinsDiscretizer}.
 *
 * @param <T> The class type of this instance.
 */
public interface KBinsDiscretizerParams<T> extends KBinsDiscretizerModelParams<T> {
    String UNIFORM = "uniform";
    String QUANTILE = "quantile";
    String KMEANS = "kmeans";

    /**
     * Supported options to define the widths of the bins are listed as follows.
     *
     * <ul>
     *   <li>uniform: all bins in each feature have identical widths.
     *   <li>quantile: all bins in each feature have the same number of points.
     *   <li>kmeans: values in each bin have the same nearest center of a 1D kmeans cluster.
     * </ul>
     */
    Param<String> STRATEGY =
            new StringParam(
                    "strategy",
                    "Strategy used to define the width of the bin.",
                    QUANTILE,
                    ParamValidators.inArray(UNIFORM, QUANTILE, KMEANS));

    Param<Integer> NUM_BINS =
            new IntParam("numBins", "Number of bins to produce.", 5, ParamValidators.gtEq(2));

    Param<Integer> SUB_SAMPLES =
            new IntParam(
                    "subSamples",
                    "Maximum number of samples used to fit the model.",
                    200000,
                    ParamValidators.gtEq(2));

    default String getStrategy() {
        return get(STRATEGY);
    }

    default T setStrategy(String value) {
        return set(STRATEGY, value);
    }

    default int getNumBins() {
        return get(NUM_BINS);
    }

    default T setNumBins(int value) {
        return set(NUM_BINS, value);
    }

    default int getSubSamples() {
        return get(SUB_SAMPLES);
    }

    default T setSubSamples(Integer value) {
        return set(SUB_SAMPLES, value);
    }
}
