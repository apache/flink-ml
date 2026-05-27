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

package org.apache.flink.ml.common.gbt.operators;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.TaskType;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.lossfunc.LogLoss;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.common.lossfunc.SquaredErrorLoss;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

class TrainContextInitializer {
    private static final Logger LOG = LoggerFactory.getLogger(TrainContextInitializer.class);
    private final BoostingStrategy strategy;

    public TrainContextInitializer(BoostingStrategy strategy) {
        this.strategy = strategy;
    }

    /**
     * Initializes local state.
     *
     * <p>Note that local state already has some properties set in advance, see GBTRunner#boost.
     */
    public TrainContext init(
            TrainContext trainContext, int subtaskId, int numSubtasks, BinnedInstance[] instances) {
        LOG.info(
                "subtaskId: {}, {} start",
                subtaskId,
                TrainContextInitializer.class.getSimpleName());

        trainContext.subtaskId = subtaskId;
        trainContext.numSubtasks = numSubtasks;

        int numInstances = instances.length;
        int numFeatures = trainContext.featureMetas.length;

        LOG.info(
                "subtaskId: {}, #samples: {}, #features: {}", subtaskId, numInstances, numFeatures);

        trainContext.strategy = strategy;
        trainContext.numInstances = numInstances;
        trainContext.numFeatures = numFeatures;

        trainContext.numBaggingInstances = getNumBaggingSamples(numInstances);
        trainContext.numBaggingFeatures = getNumBaggingFeatures(numFeatures);

        trainContext.instanceRandomizer = new Random(subtaskId + strategy.seed);
        trainContext.featureRandomizer = new Random(strategy.seed);

        trainContext.loss = getLoss();
        trainContext.prior = calcPrior(trainContext.labelSumCount);

        // A special `feature` is appended with #bins = 1 to simplify codes.
        trainContext.numFeatureBins =
                ArrayUtils.add(
                        Arrays.stream(trainContext.featureMetas)
                                .mapToInt(d -> d.numBins(trainContext.strategy.useMissing))
                                .toArray(),
                        1);
        LOG.debug("Number of bins for each feature: {}", trainContext.numFeatureBins);
        LOG.info("subtaskId: {}, {} end", subtaskId, TrainContextInitializer.class.getSimpleName());
        return trainContext;
    }

    private int getNumBaggingSamples(int numSamples) {
        return (int) Math.min(numSamples, Math.ceil(numSamples * strategy.subsamplingRate));
    }

    private int getNumBaggingFeatures(int numFeatures) {
        final List<String> supported = Arrays.asList("auto", "all", "onethird", "sqrt", "log2");
        final String errorMsg =
                String.format(
                        "Parameter `featureSubsetStrategy` supports %s, (0.0 - 1.0], [1 - n].",
                        String.join(", ", supported));
        final Function<Double, Integer> clamp =
                d -> Math.max(1, Math.min(d.intValue(), numFeatures));
        String featureSubsetStrategy = strategy.featureSubsetStrategy;
        try {
            int numBaggingFeatures = Integer.parseInt(featureSubsetStrategy);
            Preconditions.checkArgument(
                    numBaggingFeatures >= 1 && numBaggingFeatures <= numFeatures, errorMsg);
        } catch (NumberFormatException ignored) {
        }
        try {
            double baggingRatio = Double.parseDouble(featureSubsetStrategy);
            Preconditions.checkArgument(baggingRatio > 0. && baggingRatio <= 1., errorMsg);
            return clamp.apply(baggingRatio * numFeatures);
        } catch (NumberFormatException ignored) {
        }

        Preconditions.checkArgument(supported.contains(featureSubsetStrategy), errorMsg);
        switch (featureSubsetStrategy) {
            case "auto":
                return TaskType.CLASSIFICATION.equals(strategy.taskType)
                        ? clamp.apply(Math.sqrt(numFeatures))
                        : clamp.apply(numFeatures / 3.);
            case "all":
                return numFeatures;
            case "onethird":
                return clamp.apply(numFeatures / 3.);
            case "sqrt":
                return clamp.apply(Math.sqrt(numFeatures));
            case "log2":
                return clamp.apply(Math.log(numFeatures) / Math.log(2));
            default:
                throw new IllegalArgumentException(errorMsg);
        }
    }

    private LossFunc getLoss() {
        switch (strategy.lossType) {
            case LOGISTIC:
                return LogLoss.INSTANCE;
            case SQUARED:
                return SquaredErrorLoss.INSTANCE;
            default:
                throw new UnsupportedOperationException("Unsupported loss.");
        }
    }

    private double calcPrior(Tuple2<Double, Long> labelStat) {
        switch (strategy.lossType) {
            case LOGISTIC:
                return Math.log(labelStat.f0 / (labelStat.f1 - labelStat.f0));
            case SQUARED:
                return labelStat.f0 / labelStat.f1;
            default:
                throw new UnsupportedOperationException("Unsupported loss.");
        }
    }
}
