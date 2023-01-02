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

import org.apache.flink.ml.param.LongParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WithParams;

/** Interface for the shared max allowed model delay in milliseconds param. */
public interface HasMaxAllowedModelDelayMs<T> extends WithParams<T> {
    Param<Long> MAX_ALLOWED_MODEL_DELAY_MS =
            new LongParam(
                    "maxAllowedModelDelayMs",
                    "The maximum difference allowed between the timestamps of the input record "
                            + "and the model data that is used to predict that input record. "
                            + "This param only works when the input contains event time.",
                    0L,
                    ParamValidators.gtEq(0));

    default long getMaxAllowedModelDelayMs() {
        return get(MAX_ALLOWED_MODEL_DELAY_MS);
    }

    default T setMaxAllowedModelDelayMs(long value) {
        return set(MAX_ALLOWED_MODEL_DELAY_MS, value);
    }
}
