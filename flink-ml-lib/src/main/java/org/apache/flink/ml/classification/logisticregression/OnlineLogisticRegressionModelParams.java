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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.common.param.HasRawPredictionCol;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link OnlineLogisticRegressionModel}.
 *
 * @param <T> The class type of this instance.
 */
public interface OnlineLogisticRegressionModelParams<T>
        extends HasFeaturesCol<T>, HasPredictionCol<T>, HasRawPredictionCol<T> {
    Param<String> MODEL_VERSION_COL =
            new StringParam(
                    "modelVersionCol",
                    "Model version column name.",
                    "modelVersion",
                    ParamValidators.notNull());

    default String getModelVersionCol() {
        return get(MODEL_VERSION_COL);
    }

    default T setModelVersionCol(String value) {
        set(MODEL_VERSION_COL, value);
        return (T) this;
    }
}
