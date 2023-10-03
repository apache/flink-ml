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

/** Splitter for a continuous feature. */
public final class ContinuousFeatureSplitter extends HistogramFeatureSplitter {

    public ContinuousFeatureSplitter(
            int featureId, FeatureMeta featureMeta, BoostingStrategy strategy) {
        super(featureId, featureMeta, strategy);
    }

    @Override
    public Split.ContinuousSplit bestSplit() {
        HessianImpurity total = emptyImpurity();
        HessianImpurity missing = emptyImpurity();
        countTotalMissing(total, missing);

        if (total.getNumInstances() <= minSamplesPerLeaf) {
            return Split.ContinuousSplit.invalid(total.prediction());
        }

        Tuple3<Double, Integer, Boolean> bestSplit = findBestSplit(slice.size(), total, missing);
        double bestGain = bestSplit.f0;
        int bestSplitBinId = bestSplit.f1;
        boolean missingGoLeft = bestSplit.f2;

        if (bestGain <= Split.INVALID_GAIN || bestGain <= minInfoGain) {
            return Split.ContinuousSplit.invalid(total.prediction());
        }
        int splitPoint =
                useMissing && bestSplitBinId > featureMeta.missingBin
                        ? bestSplitBinId - 1
                        : bestSplitBinId;
        return new Split.ContinuousSplit(
                featureId,
                bestGain,
                featureMeta.missingBin,
                missingGoLeft,
                total.prediction(),
                splitPoint,
                !strategy.isInputVector,
                ((FeatureMeta.ContinuousFeatureMeta) featureMeta).zeroBin);
    }
}
