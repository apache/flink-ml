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

import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.Split;

/**
 * Tests if the node can be split on a given feature and obtains best split.
 *
 * <p>When testing the node, we only check internal criteria, such as minimum info gain, minium
 * samples per leaf, etc. The external criteria, like maximum depth or maximum number of leaves are
 * not checked.
 */
public abstract class FeatureSplitter {
    protected final int featureId;
    protected final FeatureMeta featureMeta;
    protected final GbtParams params;

    protected final int minSamplesPerLeaf;
    protected final double minSampleRatioPerChild;
    protected final double minInfoGain;

    public FeatureSplitter(int featureId, FeatureMeta featureMeta, GbtParams params) {
        this.params = params;
        this.featureId = featureId;
        this.featureMeta = featureMeta;

        this.minSamplesPerLeaf = params.minInstancesPerNode;
        this.minSampleRatioPerChild = params.minWeightFractionPerNode; // TODO: not exactly the same
        this.minInfoGain = params.minInfoGain;
    }

    public abstract Split bestSplit();
}
