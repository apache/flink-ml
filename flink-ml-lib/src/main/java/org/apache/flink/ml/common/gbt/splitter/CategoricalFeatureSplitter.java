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
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.HessianImpurity;
import org.apache.flink.ml.common.gbt.defs.Split;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;

import static org.apache.flink.ml.common.gbt.DataUtils.BIN_SIZE;

/** Splitter for a categorical feature using LightGBM many-vs-many split. */
public class CategoricalFeatureSplitter extends HistogramFeatureSplitter {

    public CategoricalFeatureSplitter(int featureId, FeatureMeta featureMeta, GbtParams params) {
        super(featureId, featureMeta, params);
    }

    @Override
    public Split.CategoricalSplit bestSplit() {
        Tuple2<HessianImpurity, HessianImpurity> totalMissing = countTotalMissing();
        HessianImpurity total = totalMissing.f0;
        HessianImpurity missing = totalMissing.f1;

        if (total.getNumInstances() <= minSamplesPerLeaf) {
            return Split.CategoricalSplit.invalid(total.prediction());
        }

        int numBins = slice.size();
        // Sorts categories based on grads / hessians, i.e., LightGBM many-vs-many approach.
        Integer[] sortedCategories = new Integer[numBins];
        {
            double[] scores = new double[numBins];
            for (int i = 0; i < numBins; ++i) {
                sortedCategories[i] = i;
                int startIndex = (slice.start + i) * BIN_SIZE;
                scores[i] = hists[startIndex] / hists[startIndex + 1];
            }
            Arrays.sort(sortedCategories, Comparator.comparing(d -> scores[d]));
        }

        Tuple3<Double, Integer, Boolean> bestSplit =
                findBestSplit(ArrayUtils.toPrimitive(sortedCategories), total, missing);
        double bestGain = bestSplit.f0;
        int bestSplitBinId = bestSplit.f1;
        boolean missingGoLeft = bestSplit.f2;

        if (bestGain <= Split.INVALID_GAIN || bestGain <= minInfoGain) {
            return Split.CategoricalSplit.invalid(total.prediction());
        }

        // Indicates which bins should go left.
        BitSet binsGoLeft = new BitSet(numBins);
        if (useMissing) {
            for (int i = 0; i < numBins; ++i) {
                int binId = sortedCategories[i];
                if (i <= bestSplitBinId) {
                    if (binId < featureMeta.missingBin) {
                        binsGoLeft.set(binId);
                    } else if (binId > featureMeta.missingBin) {
                        binsGoLeft.set(binId - 1);
                    }
                }
            }
        } else {
            int numCategories =
                    ((FeatureMeta.CategoricalFeatureMeta) featureMeta).categories.length;
            for (int i = 0; i < numCategories; i += 1) {
                int binId = sortedCategories[i];
                if (i <= bestSplitBinId) {
                    binsGoLeft.set(binId);
                }
            }
        }
        return new Split.CategoricalSplit(
                featureId,
                bestGain,
                featureMeta.missingBin,
                missingGoLeft,
                total.prediction(),
                binsGoLeft);
    }
}
