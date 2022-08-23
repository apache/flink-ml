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

package org.apache.flink.ml.feature.binarizer;

import org.apache.flink.ml.common.param.HasInputCols;
import org.apache.flink.ml.common.param.HasOutputCols;
import org.apache.flink.ml.param.DoubleArrayParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params of {@link Binarizer}.
 *
 * @param <T> The class type of this instance.
 */
public interface BinarizerParams<T> extends HasInputCols<T>, HasOutputCols<T> {
    Param<Double[]> THRESHOLDS =
            new DoubleArrayParam(
                    "thresholds",
                    "The thresholds used to binarize continuous features. Each threshold would be used "
                            + "against one input column. If the value of a continuous feature is greater than the "
                            + "threshold, it will be binarized to 1.0. If the value is equal to or less than the "
                            + "threshold, it will be binarized to 0.0.",
                    null,
                    ParamValidators.nonEmptyArray());

    default Double[] getThresholds() {
        return get(THRESHOLDS);
    }

    default T setThresholds(Double... value) {
        return set(THRESHOLDS, value);
    }
}
