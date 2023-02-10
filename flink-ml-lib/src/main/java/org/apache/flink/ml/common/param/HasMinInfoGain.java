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

package org.apache.flink.ml.common.param;

import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WithParams;

/** Interface for shared param minInfoGain. */
public interface HasMinInfoGain<T> extends WithParams<T> {
    Param<Double> MIN_INFO_GAIN =
            new DoubleParam(
                    "minInfoGain",
                    "Minimum information gain for a split to be considered valid.",
                    0.,
                    ParamValidators.gtEq(0.));

    default double getMinInfoGain() {
        return get(MIN_INFO_GAIN);
    }

    default T setMinInfoGain(Double value) {
        return set(MIN_INFO_GAIN, value);
    }
}
