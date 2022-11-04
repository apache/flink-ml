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

package org.apache.flink.ml.feature.imputer;

import org.apache.flink.ml.common.param.HasRelativeError;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params of {@link Imputer}.
 *
 * @param <T> The class type of this instance.
 */
public interface ImputerParams<T> extends HasRelativeError<T>, ImputerModelParams<T> {
    String MEAN = "mean";
    String MEDIAN = "median";
    String MOST_FREQUENT = "most_frequent";

    /**
     * Supported options of the imputation strategy.
     *
     * <ul>
     *   <li>mean: replace missing values using the mean along each column.
     *   <li>median: replace missing values using the median along each column.
     *   <li>most_frequent: replace missing using the most frequent value along each column. If
     *       there is more than one such value, only the smallest is returned.
     * </ul>
     */
    Param<String> STRATEGY =
            new StringParam(
                    "strategy",
                    "The imputation strategy.",
                    MEAN,
                    ParamValidators.inArray(MEAN, MEDIAN, MOST_FREQUENT));

    default String getStrategy() {
        return get(STRATEGY);
    }

    default T setStrategy(String value) {
        return set(STRATEGY, value);
    }
}
