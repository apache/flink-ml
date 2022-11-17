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

package org.apache.flink.ml.feature.robustscaler;

import org.apache.flink.ml.common.param.HasInputCol;
import org.apache.flink.ml.common.param.HasOutputCol;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.Param;

/**
 * Params for {@link RobustScalerModel}.
 *
 * @param <T> The class type of this instance.
 */
public interface RobustScalerModelParams<T> extends HasInputCol<T>, HasOutputCol<T> {
    Param<Boolean> WITH_CENTERING =
            new BooleanParam(
                    "withCentering",
                    "Whether to center the data with median before scaling.",
                    false);

    Param<Boolean> WITH_SCALING =
            new BooleanParam("withScaling", "Whether to scale the data to quantile range.", true);

    default boolean getWithCentering() {
        return get(WITH_CENTERING);
    }

    default T setWithCentering(boolean value) {
        return set(WITH_CENTERING, value);
    }

    default boolean getWithScaling() {
        return get(WITH_SCALING);
    }

    default T setWithScaling(boolean value) {
        return set(WITH_SCALING, value);
    }
}
