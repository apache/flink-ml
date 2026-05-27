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

import org.apache.flink.ml.common.gbt.DataUtils;

import java.io.Serializable;
import java.util.Arrays;

/** Stores meta information of a feature. */
public abstract class FeatureMeta {

    public String name;
    public Type type;
    // The bin index representing the missing values.
    public int missingBin;

    public FeatureMeta() {}

    public FeatureMeta(String name, int missingBin, Type type) {
        this.name = name;
        this.missingBin = missingBin;
        this.type = type;
    }

    public static CategoricalFeatureMeta categorical(
            String name, int missingBin, String[] categories) {
        return new CategoricalFeatureMeta(name, missingBin, categories);
    }

    public static ContinuousFeatureMeta continuous(String name, int missingBin, double[] binEdges) {
        return new ContinuousFeatureMeta(name, missingBin, binEdges);
    }

    /**
     * Calculate number of bins used for this feature.
     *
     * @param useMissing Whether to assign an addition bin for missing values.
     * @return The number of bins.
     */
    public abstract int numBins(boolean useMissing);

    @Override
    public String toString() {
        return String.format(
                "FeatureMeta{name='%s', type=%s, missingBin=%d}", name, type, missingBin);
    }

    /** Indicates the feature type. */
    public enum Type implements Serializable {
        CATEGORICAL,
        CONTINUOUS
    }

    /** Stores meta information for a categorical feature. */
    public static class CategoricalFeatureMeta extends FeatureMeta {
        // Stores ordered categorical values.
        public String[] categories;

        public CategoricalFeatureMeta() {}

        public CategoricalFeatureMeta(String name, int missingBin, String[] categories) {
            super(name, missingBin, Type.CATEGORICAL);
            this.categories = categories;
        }

        @Override
        public int numBins(boolean useMissing) {
            return useMissing ? categories.length + 1 : categories.length;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            }
            return obj instanceof CategoricalFeatureMeta
                    && this.type.equals(((CategoricalFeatureMeta) obj).type)
                    && (this.name.equals(((CategoricalFeatureMeta) obj).name))
                    && (this.missingBin == ((CategoricalFeatureMeta) obj).missingBin)
                    && (Arrays.equals(this.categories, ((CategoricalFeatureMeta) obj).categories));
        }

        @Override
        public String toString() {
            return String.format(
                    "CategoricalFeatureMeta{categories=%s} %s",
                    Arrays.toString(categories), super.toString());
        }
    }

    /** Stores meta information for a continuous feature. */
    public static class ContinuousFeatureMeta extends FeatureMeta {
        // Stores the edges of bins.
        public double[] binEdges;
        // The bin index for value 0.
        public int zeroBin;

        public ContinuousFeatureMeta() {}

        public ContinuousFeatureMeta(String name, int missingBin, double[] binEdges) {
            super(name, missingBin, Type.CONTINUOUS);
            this.binEdges = binEdges;
            this.zeroBin = DataUtils.findBin(binEdges, 0.);
        }

        @Override
        public int numBins(boolean useMissing) {
            return useMissing ? binEdges.length : binEdges.length - 1;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == this) {
                return true;
            }
            return obj instanceof ContinuousFeatureMeta
                    && this.type.equals(((ContinuousFeatureMeta) obj).type)
                    && (this.name.equals(((ContinuousFeatureMeta) obj).name))
                    && (this.missingBin == ((ContinuousFeatureMeta) obj).missingBin)
                    && (Arrays.equals(this.binEdges, ((ContinuousFeatureMeta) obj).binEdges))
                    && (this.zeroBin == ((ContinuousFeatureMeta) obj).zeroBin);
        }

        @Override
        public String toString() {
            return String.format(
                    "ContinuousFeatureMeta{binEdges=%s, zeroBin=%d} %s",
                    Arrays.toString(binEdges), zeroBin, super.toString());
        }
    }
}
