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

import org.apache.flink.ml.common.param.HasElasticNet;
import org.apache.flink.ml.common.param.HasGlobalBatchSize;
import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasMaxIter;
import org.apache.flink.ml.common.param.HasMultiClass;
import org.apache.flink.ml.common.param.HasReg;
import org.apache.flink.ml.common.param.HasTol;
import org.apache.flink.ml.common.param.HasWeightCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.LongParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/** Params for {@link LogisticRegressionWithFtrl}. */
public interface LogisticRegressionWithFtrlParams<T>
        extends HasLabelCol<T>,
                HasWeightCol<T>,
                HasGlobalBatchSize<T>,
                HasReg<T>,
                HasElasticNet<T>,
                HasMultiClass<T>,
                HasMaxIter<T>,
                HasTol<T>,
                LogisticRegressionModelParams<T> {

    Param<Integer> NUM_SERVERS =
            new IntParam(
                    "numServers",
                    "Number of servers to store model parameters.",
                    1,
                    ParamValidators.gtEq(1));

    Param<Double> ALPHA =
            new DoubleParam(
                    "alpha",
                    "The alpha parameter of FTRL optimizer.",
                    0.1,
                    ParamValidators.gt(0.0));

    Param<Double> BETA =
            new DoubleParam(
                    "beta", "The beta parameter of FTRL optimizer.", 0.1, ParamValidators.gt(0.0));

    Param<Long> MODEL_DIM =
            new LongParam(
                    "modelDim", "number of features of input data.", 0L, ParamValidators.gtEq(0));

    default int getNumServers() {
        return get(NUM_SERVERS);
    }

    default T setNumServers(Integer value) {
        return set(NUM_SERVERS, value);
    }

    default double getAlpha() {
        return get(ALPHA);
    }

    default T setAlpha(Double value) {
        return set(ALPHA, value);
    }

    default double getBeta() {
        return get(BETA);
    }

    default T setBeta(Double value) {
        return set(BETA, value);
    }

    default long getModelDim() {
        return get(MODEL_DIM);
    }

    default T setModelDim(long value) {
        return set(MODEL_DIM, value);
    }
}
