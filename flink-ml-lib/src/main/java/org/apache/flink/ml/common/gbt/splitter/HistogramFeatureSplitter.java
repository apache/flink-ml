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

package org.apache.flink.ml.common.gbt.splitter;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.HessianImpurity;
import org.apache.flink.ml.common.gbt.defs.Impurity;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.Split;

import static org.apache.flink.ml.common.gbt.DataUtils.BIN_SIZE;

/** Histogram based feature splitter. */
public abstract class HistogramFeatureSplitter extends FeatureSplitter {
    protected final boolean useMissing;
    protected double[] hists;
    protected Slice slice;

    public HistogramFeatureSplitter(
            int featureId, FeatureMeta featureMeta, BoostingStrategy strategy) {
        super(featureId, featureMeta, strategy);
        this.useMissing = strategy.useMissing;
    }

    protected boolean isSplitIllegal(Impurity total, Impurity left, Impurity right) {
        return (minSamplesPerLeaf > left.getTotalWeights()
                        || minSamplesPerLeaf > right.getTotalWeights())
                || minSampleRatioPerChild > 1. * left.getNumInstances() / total.getNumInstances()
                || minSampleRatioPerChild > 1. * right.getNumInstances() / total.getNumInstances();
    }

    protected double gain(Impurity total, Impurity left, Impurity right) {
        return isSplitIllegal(total, left, right) ? Split.INVALID_GAIN : total.gain(left, right);
    }

    protected void addBinToLeft(int binId, HessianImpurity left, HessianImpurity right) {
        int index = (slice.start + binId) * BIN_SIZE;
        left.add((int) hists[index + 3], hists[index + 2], hists[index], hists[index + 1]);
        if (null != right) {
            right.subtract(
                    (int) hists[index + 3], hists[index + 2], hists[index], hists[index + 1]);
        }
    }

    protected Tuple2<Double, Integer> findBestSplitWithInitial(
            int[] sortedBinIds,
            HessianImpurity total,
            HessianImpurity left,
            HessianImpurity right) {
        // Bins [0, bestSplitBinId] go left.
        int bestSplitBinId = 0;
        double bestGain = Split.INVALID_GAIN;
        for (int i = 0; i < sortedBinIds.length; i += 1) {
            int binId = sortedBinIds[i];
            if (useMissing && binId == featureMeta.missingBin) {
                continue;
            }
            addBinToLeft(binId, left, right);
            double gain = gain(total, left, right);
            if (gain > bestGain && gain >= minInfoGain) {
                bestGain = gain;
                bestSplitBinId = i;
            }
        }
        return Tuple2.of(bestGain, bestSplitBinId);
    }

    protected Tuple2<Double, Integer> findBestSplitWithInitial(
            int numBins, HessianImpurity total, HessianImpurity left, HessianImpurity right) {
        // Bins [0, bestSplitBinId] go left.
        int bestSplitBinId = 0;
        double bestGain = Split.INVALID_GAIN;
        for (int binId = 0; binId < numBins; binId += 1) {
            if (useMissing && binId == featureMeta.missingBin) {
                continue;
            }
            addBinToLeft(binId, left, right);
            double gain = gain(total, left, right);
            if (gain > bestGain && gain >= minInfoGain) {
                bestGain = gain;
                bestSplitBinId = binId;
            }
        }
        return Tuple2.of(bestGain, bestSplitBinId);
    }

    protected Tuple3<Double, Integer, Boolean> findBestSplit(
            int[] sortedBinIds, HessianImpurity total, HessianImpurity missing) {
        double bestGain = Split.INVALID_GAIN;
        int bestSplitBinId = 0;
        boolean missingGoLeft = false;

        {
            // The cases where the missing values go right, or missing values are not allowed.
            HessianImpurity left = emptyImpurity();
            HessianImpurity right = (HessianImpurity) total.clone();
            Tuple2<Double, Integer> bestSplit =
                    findBestSplitWithInitial(sortedBinIds, total, left, right);
            if (bestSplit.f0 > bestGain) {
                bestGain = bestSplit.f0;
                bestSplitBinId = bestSplit.f1;
            }
        }

        if (useMissing && missing.getNumInstances() > 0) {
            // The cases where the missing values go left.
            HessianImpurity leftWithMissing = emptyImpurity().add(missing);
            HessianImpurity rightWithoutMissing = (HessianImpurity) total.clone().subtract(missing);
            Tuple2<Double, Integer> bestSplitMissingGoLeft =
                    findBestSplitWithInitial(
                            sortedBinIds, total, leftWithMissing, rightWithoutMissing);
            if (bestSplitMissingGoLeft.f0 > bestGain) {
                bestGain = bestSplitMissingGoLeft.f0;
                bestSplitBinId = bestSplitMissingGoLeft.f1;
                missingGoLeft = true;
            }
        }
        return Tuple3.of(bestGain, bestSplitBinId, missingGoLeft);
    }

    protected Tuple3<Double, Integer, Boolean> findBestSplit(
            int numBins, HessianImpurity total, HessianImpurity missing) {
        double bestGain = Split.INVALID_GAIN;
        int bestSplitBinId = 0;
        boolean missingGoLeft = false;

        {
            // The cases where the missing values go right, or missing values are not allowed.
            HessianImpurity left = emptyImpurity();
            HessianImpurity right = (HessianImpurity) total.clone();
            Tuple2<Double, Integer> bestSplit =
                    findBestSplitWithInitial(numBins, total, left, right);
            if (bestSplit.f0 > bestGain) {
                bestGain = bestSplit.f0;
                bestSplitBinId = bestSplit.f1;
            }
        }

        if (useMissing) {
            // The cases where the missing values go left.
            HessianImpurity leftWithMissing = emptyImpurity().add(missing);
            HessianImpurity rightWithoutMissing = (HessianImpurity) total.clone().subtract(missing);
            Tuple2<Double, Integer> bestSplitMissingGoLeft =
                    findBestSplitWithInitial(numBins, total, leftWithMissing, rightWithoutMissing);
            if (bestSplitMissingGoLeft.f0 > bestGain) {
                bestGain = bestSplitMissingGoLeft.f0;
                bestSplitBinId = bestSplitMissingGoLeft.f1;
                missingGoLeft = true;
            }
        }
        return Tuple3.of(bestGain, bestSplitBinId, missingGoLeft);
    }

    public void reset(double[] hists, Slice slice) {
        this.hists = hists;
        this.slice = slice;
    }

    protected void countTotalMissing(HessianImpurity total, HessianImpurity missing) {
        for (int i = 0; i < slice.size(); ++i) {
            addBinToLeft(i, total, null);
        }
        if (useMissing) {
            addBinToLeft(featureMeta.missingBin, missing, null);
        }
    }

    protected HessianImpurity emptyImpurity() {
        return new HessianImpurity(strategy.regLambda, strategy.regGamma, 0, 0, 0, 0);
    }
}
