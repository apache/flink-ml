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

import org.apache.flink.ml.common.param.HasBatchStrategy;
import org.apache.flink.ml.common.param.HasElasticNet;
import org.apache.flink.ml.common.param.HasGlobalBatchSize;
import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasReg;
import org.apache.flink.ml.common.param.HasWeightCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params of {@link OnlineLogisticRegression}.
 *
 * @param <T> The class type of this instance.
 */
public interface OnlineLogisticRegressionParams<T>
        extends HasLabelCol<T>,
                HasWeightCol<T>,
                HasBatchStrategy<T>,
                HasGlobalBatchSize<T>,
                HasReg<T>,
                HasElasticNet<T>,
                OnlineLogisticRegressionModelParams<T> {

    Param<Double> ALPHA =
            new DoubleParam("alpha", "The alpha parameter of ftrl.", 0.1, ParamValidators.gt(0.0));

    Param<Double> BETA =
            new DoubleParam("beta", "The beta parameter of ftrl.", 0.1, ParamValidators.gt(0.0));

    default Double getAlpha() {
        return get(ALPHA);
    }

    default T setAlpha(Double value) {
        return set(ALPHA, value);
    }

    default Double getBeta() {
        return get(BETA);
    }

    default T setBeta(Double value) {
        return set(BETA, value);
    }
}
