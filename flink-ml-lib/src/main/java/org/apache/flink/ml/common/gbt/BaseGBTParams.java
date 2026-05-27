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

import org.apache.flink.ml.common.param.HasMaxIter;
import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.common.param.HasWeightCol;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Common parameters for GBT classifier and regressor.
 *
 * <p>NOTE: Features related to {@link #WEIGHT_COL}, {@link #LEAF_COL}, and {@link
 * #VALIDATION_INDICATOR_COL} are not implemented yet.
 *
 * @param <T> The class type of this instance.
 */
public interface BaseGBTParams<T>
        extends BaseGBTModelParams<T>, HasWeightCol<T>, HasMaxIter<T>, HasSeed<T> {
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

    Param<String> LEAF_COL =
            new StringParam("leafCol", "Predicted leaf index of each instance in each tree.", null);

    Param<Integer> MAX_DEPTH =
            new IntParam("maxDepth", "Maximum depth of the tree.", 5, ParamValidators.gtEq(1));

    Param<Integer> MAX_BINS =
            new IntParam(
                    "maxBins",
                    "Maximum number of bins used for discretizing continuous features.",
                    32,
                    ParamValidators.gtEq(2));

    Param<Integer> MIN_INSTANCES_PER_NODE =
            new IntParam(
                    "minInstancesPerNode",
                    "Minimum number of instances each node must have. If a split causes the left or right child to have fewer instances than minInstancesPerNode, the split is invalid.",
                    1,
                    ParamValidators.gtEq(1));

    Param<Double> MIN_WEIGHT_FRACTION_PER_NODE =
            new DoubleParam(
                    "minWeightFractionPerNode",
                    "Minimum fraction of the weighted sample count that each node must have. If a split causes the left or right child to have a smaller fraction of the total weight than minWeightFractionPerNode, the split is invalid.",
                    0.,
                    ParamValidators.gtEq(0.));

    Param<Double> MIN_INFO_GAIN =
            new DoubleParam(
                    "minInfoGain",
                    "Minimum information gain for a split to be considered valid.",
                    0.,
                    ParamValidators.gtEq(0.));

    Param<Double> STEP_SIZE =
            new DoubleParam(
                    "stepSize",
                    "Step size for shrinking the contribution of each estimator.",
                    0.1,
                    ParamValidators.inRange(0., 1.));

    Param<Double> SUBSAMPLING_RATE =
            new DoubleParam(
                    "subsamplingRate",
                    "Fraction of the training data used for learning one tree.",
                    1.,
                    ParamValidators.inRange(0., 1.));

    Param<String> FEATURE_SUBSET_STRATEGY =
            new StringParam(
                    "featureSubsetStrategy.",
                    "Fraction of the training data used for learning one tree. Supports \"auto\", \"all\", \"onethird\", \"sqrt\", \"log2\", (0.0 - 1.0], and [1 - n].",
                    "auto",
                    ParamValidators.notNull());

    Param<String> VALIDATION_INDICATOR_COL =
            new StringParam(
                    "validationIndicatorCol",
                    "The name of the column that indicates whether each row is for training or for validation.",
                    null);

    Param<Double> VALIDATION_TOL =
            new DoubleParam(
                    "validationTol",
                    "Threshold for early stopping when fitting with validation is used.",
                    .01,
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

    default String getLeafCol() {
        return get(LEAF_COL);
    }

    default T setLeafCol(String value) {
        return set(LEAF_COL, value);
    }

    default int getMaxDepth() {
        return get(MAX_DEPTH);
    }

    default T setMaxDepth(int value) {
        return set(MAX_DEPTH, value);
    }

    default int getMaxBins() {
        return get(MAX_BINS);
    }

    default T setMaxBins(int value) {
        return set(MAX_BINS, value);
    }

    default int getMinInstancesPerNode() {
        return get(MIN_INSTANCES_PER_NODE);
    }

    default T setMinInstancesPerNode(int value) {
        return set(MIN_INSTANCES_PER_NODE, value);
    }

    default double getMinWeightFractionPerNode() {
        return get(MIN_WEIGHT_FRACTION_PER_NODE);
    }

    default T setMinWeightFractionPerNode(Double value) {
        return set(MIN_WEIGHT_FRACTION_PER_NODE, value);
    }

    default double getMinInfoGain() {
        return get(MIN_INFO_GAIN);
    }

    default T setMinInfoGain(Double value) {
        return set(MIN_INFO_GAIN, value);
    }

    default double getStepSize() {
        return get(STEP_SIZE);
    }

    default T setStepSize(Double value) {
        return set(STEP_SIZE, value);
    }

    default double getSubsamplingRate() {
        return get(SUBSAMPLING_RATE);
    }

    default T setSubsamplingRate(Double value) {
        return set(SUBSAMPLING_RATE, value);
    }

    default String getFeatureSubsetStrategy() {
        return get(FEATURE_SUBSET_STRATEGY);
    }

    default T setFeatureSubsetStrategy(String value) {
        return set(FEATURE_SUBSET_STRATEGY, value);
    }

    default String getValidationIndicatorCol() {
        return get(VALIDATION_INDICATOR_COL);
    }

    default T setValidationIndicatorCol(String value) {
        return set(VALIDATION_INDICATOR_COL, value);
    }

    default double getValidationTol() {
        return get(VALIDATION_TOL);
    }

    default T setValidationTol(Double value) {
        return set(VALIDATION_TOL, value);
    }
}
