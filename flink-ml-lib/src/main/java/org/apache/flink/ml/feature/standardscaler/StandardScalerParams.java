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

package org.apache.flink.ml.feature.standardscaler;

import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.Param;

/**
 * Params for {@link StandardScaler}.
 *
 * @param <T> The class type of this instance.
 */
public interface StandardScalerParams<T> extends HasFeaturesCol<T>, HasPredictionCol<T> {
    Param<Boolean> WITH_MEAN =
            new BooleanParam(
                    "withMean", "Whether centers the data with mean before scaling.", false);

    Param<Boolean> WITH_STD =
            new BooleanParam("withStd", "Whether scales the data with standard deviation.", true);

    default Boolean getWithMean() {
        return get(WITH_MEAN);
    }

    default T setWithMean(boolean withMean) {
        return set(WITH_MEAN, withMean);
    }

    default Boolean getWithStd() {
        return get(WITH_STD);
    }

    default T setWithStd(boolean withMean) {
        return set(WITH_STD, withMean);
    }
}
