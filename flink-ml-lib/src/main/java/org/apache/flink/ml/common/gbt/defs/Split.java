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

package org.apache.flink.ml.common.gbt.defs;

import org.apache.flink.api.common.typeinfo.TypeInfo;
import org.apache.flink.ml.common.gbt.typeinfo.SplitTypeInfoFactory;

import org.eclipse.collections.impl.map.mutable.primitive.IntDoubleHashMap;

import java.util.BitSet;

/** Stores a split on a feature. */
@TypeInfo(SplitTypeInfoFactory.class)
public abstract class Split {
    public static final double INVALID_GAIN = 0.0;

    // Stores the feature index of this split.
    public final int featureId;

    // Stores impurity gain. A value of `INVALID_GAIN` indicates this split is invalid.
    public final double gain;

    // Bin index for missing values of this feature.
    public final int missingBin;
    // Whether the missing values should go left.
    public final boolean missingGoLeft;

    // The prediction value if this split is invalid.
    public final double prediction;

    public Split(
            int featureId, double gain, int missingBin, boolean missingGoLeft, double prediction) {
        this.featureId = featureId;
        this.gain = gain;
        this.missingBin = missingBin;
        this.missingGoLeft = missingGoLeft;
        this.prediction = prediction;
    }

    public Split accumulate(Split other) {
        if (gain < other.gain) {
            return other;
        } else if (gain == other.gain) {
            if (featureId < other.featureId) {
                return other;
            }
        }
        return this;
    }

    /**
     * Test the binned instance should go to the left child or the right child.
     *
     * @param binnedInstance The instance after binned.
     * @return True if the instance should go to the left child.
     */
    public abstract boolean shouldGoLeft(BinnedInstance binnedInstance);

    /**
     * Test the raw features should go to the left child or the right child. In the raw features,
     * the categorical values are mapped to integers, while the continuous values are kept unmapped.
     *
     * @param rawFeatures The feature map from feature indices to values.
     * @return True if the raw features should go to the left child.
     */
    public abstract boolean shouldGoLeft(IntDoubleHashMap rawFeatures);

    public boolean isValid() {
        return gain != INVALID_GAIN;
    }

    /** Stores a split on a continuous feature. */
    public static class ContinuousSplit extends Split {

        /**
         * Stores the threshold that one continuous feature should go the left or right. Before
         * splitting the node, the threshold is the bin index. After that, the threshold is replaced
         * with the actual value of the bin edge.
         */
        public double threshold;

        // True if treat unseen values as missing values, otherwise treat them as 0s.
        public boolean isUnseenMissing;

        // Bin index for 0 values.
        public int zeroBin;

        public ContinuousSplit(
                int featureIndex,
                double gain,
                int missingBin,
                boolean missingGoLeft,
                double prediction,
                double threshold,
                boolean isUnseenMissing,
                int zeroBin) {
            super(featureIndex, gain, missingBin, missingGoLeft, prediction);
            this.threshold = threshold;
            this.isUnseenMissing = isUnseenMissing;
            this.zeroBin = zeroBin;
        }

        public static ContinuousSplit invalid(double prediction) {
            return new ContinuousSplit(0, INVALID_GAIN, 0, false, prediction, 0., false, 0);
        }

        @Override
        public boolean shouldGoLeft(BinnedInstance binnedInstance) {
            int index = binnedInstance.getFeatureIndex(featureId);
            if (index < 0 && isUnseenMissing) {
                return missingGoLeft;
            }
            int binId = index >= 0 ? binnedInstance.featureValues[index] : zeroBin;
            return binId == missingBin ? missingGoLeft : binId <= threshold;
        }

        @Override
        public boolean shouldGoLeft(IntDoubleHashMap rawFeatures) {
            if (!rawFeatures.containsKey(featureId) && isUnseenMissing) {
                return missingGoLeft;
            }
            double v = rawFeatures.getIfAbsent(featureId, 0.);
            return Double.isNaN(v) ? missingGoLeft : v < threshold;
        }
    }

    /** Stores a split on a categorical feature. */
    public static class CategoricalSplit extends Split {
        // Stores the indices of categorical values that should go to the left child.
        public final BitSet categoriesGoLeft;

        public CategoricalSplit(
                int featureId,
                double gain,
                int missingBin,
                boolean missingGoLeft,
                double prediction,
                BitSet categoriesGoLeft) {
            super(featureId, gain, missingBin, missingGoLeft, prediction);
            this.categoriesGoLeft = categoriesGoLeft;
        }

        public static CategoricalSplit invalid(double prediction) {
            return new CategoricalSplit(0, INVALID_GAIN, 0, false, prediction, new BitSet());
        }

        @Override
        public boolean shouldGoLeft(BinnedInstance binnedInstance) {
            int index = binnedInstance.getFeatureIndex(featureId);
            if (index < 0) {
                return missingGoLeft;
            }
            int binId = binnedInstance.featureValues[index];
            return binId == missingBin ? missingGoLeft : categoriesGoLeft.get(binId);
        }

        @Override
        public boolean shouldGoLeft(IntDoubleHashMap rawFeatures) {
            if (!rawFeatures.containsKey(featureId)) {
                return missingGoLeft;
            }
            return categoriesGoLeft.get((int) rawFeatures.get(featureId));
        }
    }
}
