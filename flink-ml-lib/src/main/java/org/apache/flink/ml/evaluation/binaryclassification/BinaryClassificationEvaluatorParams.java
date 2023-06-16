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

package org.apache.flink.ml.evaluation.binaryclassification;

import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasRawPredictionCol;
import org.apache.flink.ml.common.param.HasWeightCol;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringArrayParam;

/**
 * Params of BinaryClassificationEvaluator.
 *
 * @param <T> The class type of this instance.
 */
public interface BinaryClassificationEvaluatorParams<T>
        extends HasLabelCol<T>, HasRawPredictionCol<T>, HasWeightCol<T> {
    String AREA_UNDER_ROC = "areaUnderROC";
    String AREA_UNDER_PR = "areaUnderPR";
    String AREA_UNDER_LORENZ = "areaUnderLorenz";
    String KS = "ks";
    String PRECISION = "precision";
    String RECALL = "recall";
    String F1 = "f1";

    /**
     * Param for supported metric names in binary classification evaluation (supports
     * 'areaUnderROC', 'areaUnderPR', 'ks' and 'areaUnderLorenz').
     *
     * <ul>
     *   <li>areaUnderROC: the area under the receiver operating characteristic (ROC) curve.
     *   <li>areaUnderPR: the area under the precision-recall curve.
     *   <li>ks: Kolmogorov-Smirnov, measures the ability of the model to separate positive and
     *       negative samples.
     *   <li>areaUnderLorenz: the area under the lorenz curve.
     * </ul>
     */
    Param<String[]> METRICS_NAMES =
            new StringArrayParam(
                    "metricsNames",
                    "Names of output metrics.",
                    new String[] {AREA_UNDER_ROC, AREA_UNDER_PR},
                    ParamValidators.isSubSet(AREA_UNDER_ROC, AREA_UNDER_PR, KS, AREA_UNDER_LORENZ));

    default String[] getMetricsNames() {
        return get(METRICS_NAMES);
    }

    default T setMetricsNames(String... value) {
        return set(METRICS_NAMES, value);
    }
}
