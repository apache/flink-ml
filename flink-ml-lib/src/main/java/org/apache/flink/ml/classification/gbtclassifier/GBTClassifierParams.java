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

package org.apache.flink.ml.classification.gbtclassifier;

import org.apache.flink.ml.common.gbt.BaseGBTParams;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Parameters for {@link GBTClassifier}.
 *
 * @param <T> The class type of this instance.
 */
public interface GBTClassifierParams<T> extends BaseGBTParams<T>, GBTClassifierModelParams<T> {

    Param<String> LOSS_TYPE =
            new StringParam(
                    "lossType", "Loss type.", "logistic", ParamValidators.inArray("logistic"));

    default String getLossType() {
        return get(LOSS_TYPE);
    }

    default T setLossType(String value) {
        return set(LOSS_TYPE, value);
    }
}
