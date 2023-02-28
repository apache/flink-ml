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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.ml.common.param.HasFeatureSubsetStrategy;
import org.apache.flink.ml.common.param.HasLeafCol;
import org.apache.flink.ml.common.param.HasMaxBins;
import org.apache.flink.ml.common.param.HasMaxDepth;
import org.apache.flink.ml.common.param.HasMaxIter;
import org.apache.flink.ml.common.param.HasMinInfoGain;
import org.apache.flink.ml.common.param.HasMinInstancesPerNode;
import org.apache.flink.ml.common.param.HasMinWeightFractionPerNode;
import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.common.param.HasStepSize;
import org.apache.flink.ml.common.param.HasSubsamplingRate;
import org.apache.flink.ml.common.param.HasValidationIndicatorCol;
import org.apache.flink.ml.common.param.HasValidationTol;
import org.apache.flink.ml.common.param.HasWeightCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Common parameters for GBT classifier and regressor.
 *
 * <p>TODO: support param thresholds, impurity (actually meaningless)
 *
 * @param <T> The class type of this instance.
 */
public interface BaseGBTParams<T>
        extends BaseGBTModelParams<T>,
                HasLeafCol<T>,
                HasWeightCol<T>,
                HasMaxDepth<T>,
                HasMaxBins<T>,
                HasMinInstancesPerNode<T>,
                HasMinWeightFractionPerNode<T>,
                HasMinInfoGain<T>,
                HasMaxIter<T>,
                HasStepSize<T>,
                HasSeed<T>,
                HasSubsamplingRate<T>,
                HasFeatureSubsetStrategy<T>,
                HasValidationIndicatorCol<T>,
                HasValidationTol<T> {
    Param<Double> REG_LAMBDA =
            new DoubleParam(
                    "regLambda",
                    "Regularization term for the number of leaves.",
                    0.,
                    ParamValidators.gtEq(0.));
    Param<Double> REG_GAMMA =
            new DoubleParam(
                    "regGamma",
                    "L2 regularization term for the weights of leaves.",
                    1.,
                    ParamValidators.gtEq(0));

    default double getRegLambda() {
        return get(REG_LAMBDA);
    }

    default T setRegLambda(Double value) {
        return set(REG_LAMBDA, value);
    }

    default double getRegGamma() {
        return get(REG_GAMMA);
    }

    default T setRegGamma(Double value) {
        return set(REG_GAMMA, value);
    }
}
