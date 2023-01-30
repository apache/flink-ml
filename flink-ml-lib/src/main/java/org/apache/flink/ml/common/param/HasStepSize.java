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

/** Interface for the shared step size param. */
public interface HasStepSize<T> extends WithParams<T> {
    Param<Double> STEP_SIZE =
            new DoubleParam(
                    "stepSize",
                    "Step size for shrinking the contribution of each estimator.",
                    0.1,
                    ParamValidators.inRange(0., 1.));

    default double getStepSize() {
        return get(STEP_SIZE);
    }

    default T setStepSize(Double value) {
        return set(STEP_SIZE, value);
    }
}
