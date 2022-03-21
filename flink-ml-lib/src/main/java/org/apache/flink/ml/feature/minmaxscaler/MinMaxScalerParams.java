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

package org.apache.flink.ml.feature.minmaxscaler;

import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params for {@link MinMaxScaler}.
 *
 * @param <T> The class type of this instance.
 */
public interface MinMaxScalerParams<T> extends HasFeaturesCol<T>, HasPredictionCol<T> {
    Param<Double> MIN =
            new DoubleParam(
                    "min",
                    "Lower bound of the output feature range.",
                    0.0,
                    ParamValidators.notNull());

    default Double getMin() {
        return get(MIN);
    }

    default T setMin(Double value) {
        return set(MIN, value);
    }

    Param<Double> MAX =
            new DoubleParam(
                    "max",
                    "Upper bound of the output feature range.",
                    1.0,
                    ParamValidators.notNull());

    default Double getMax() {
        return get(MAX);
    }

    default T setMax(Double value) {
        return set(MAX, value);
    }
}
