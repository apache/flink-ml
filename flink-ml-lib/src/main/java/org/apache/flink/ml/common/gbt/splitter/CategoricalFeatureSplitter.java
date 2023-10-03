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

import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.HessianImpurity;
import org.apache.flink.ml.common.gbt.defs.Split;

import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;

import java.util.BitSet;

import static org.apache.flink.ml.common.gbt.DataUtils.BIN_SIZE;

/** Splitter for a categorical feature using LightGBM many-vs-many split. */
public class CategoricalFeatureSplitter extends HistogramFeatureSplitter {

    public CategoricalFeatureSplitter(
            int featureId, FeatureMeta featureMeta, BoostingStrategy strategy) {
        super(featureId, featureMeta, strategy);
    }

    @Override
    public Split.CategoricalSplit bestSplit() {
        HessianImpurity total = emptyImpurity();
        HessianImpurity missing = emptyImpurity();
        countTotalMissing(total, missing);

        if (total.getNumInstances() <= minSamplesPerLeaf) {
            return Split.CategoricalSplit.invalid(total.prediction());
        }

        int numBins = slice.size();
        // Sorts categories (bins） based on grads / hessians, i.e., LightGBM many-vs-many approach.
        MutableIntList sortedIndices = new IntArrayList(numBins);
        // A category (bin) is treated as missing values if its occurrences is smaller than a
        // threshold. Currently, the threshold is 0.
        BitSet ignoredIndices = new BitSet(numBins);
        {
            double[] scores = new double[numBins];
            for (int i = 0; i < numBins; ++i) {
                int index = (slice.start + i) * BIN_SIZE;
                if (hists[index + 3] > 0) {
                    sortedIndices.add(i);
                    scores[i] = hists[index] / hists[index + 1];
                } else {
                    ignoredIndices.set(i);
                    missing.add(
                            (int) hists[index + 3],
                            hists[index + 2],
                            hists[index],
                            hists[index + 1]);
                }
            }
            sortedIndices.sortThis(
                    (value1, value2) -> Double.compare(scores[value1], scores[value2]));
        }

        Tuple3<Double, Integer, Boolean> bestSplit =
                findBestSplit(sortedIndices.toArray(), total, missing);
        double bestGain = bestSplit.f0;
        int bestSplitIndex = bestSplit.f1;
        boolean missingGoLeft = bestSplit.f2;

        if (bestGain <= Split.INVALID_GAIN || bestGain <= minInfoGain) {
            return Split.CategoricalSplit.invalid(total.prediction());
        }

        // Indicates which bins should go left.
        BitSet binsGoLeft = new BitSet(numBins);
        if (useMissing) {
            for (int i = 0; i < sortedIndices.size(); ++i) {
                int binId = sortedIndices.get(i);
                if (i <= bestSplitIndex) {
                    if (binId < featureMeta.missingBin) {
                        binsGoLeft.set(binId);
                    } else if (binId > featureMeta.missingBin) {
                        binsGoLeft.set(binId - 1);
                    }
                }
            }
        } else {
            for (int i = 0; i < sortedIndices.size(); i += 1) {
                int binId = sortedIndices.get(i);
                if (i <= bestSplitIndex) {
                    binsGoLeft.set(binId);
                }
            }
        }
        if (missingGoLeft) {
            binsGoLeft.or(ignoredIndices);
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
