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

package org.apache.flink.ml.classification.linearsvc;

import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.common.param.HasRawPredictionCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params for {@link LinearSVCModel}.
 *
 * @param <T> The class type of this instance.
 */
public interface LinearSVCModelParams<T>
        extends HasFeaturesCol<T>, HasPredictionCol<T>, HasRawPredictionCol<T> {
    /**
     * Param for threshold in linear support vector classifier. It applies to the rawPrediction and
     * can be any real number, where Inf makes all predictions 0.0 and -Inf makes all predictions
     * 1.0.
     */
    Param<Double> THRESHOLD =
            new DoubleParam(
                    "threshold",
                    "Threshold in binary classification prediction applied to rawPrediction.",
                    0.0,
                    ParamValidators.notNull());

    default Double getThreshold() {
        return get(THRESHOLD);
    }

    default T setThreshold(Double value) {
        set(THRESHOLD, value);
        return (T) this;
    }
}
