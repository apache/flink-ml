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

import org.apache.flink.ml.common.gbt.defs.Distributor;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.splitter.CategoricalFeatureSplitter;
import org.apache.flink.ml.common.gbt.splitter.ContinuousFeatureSplitter;
import org.apache.flink.ml.common.gbt.splitter.HistogramFeatureSplitter;
import org.apache.flink.util.Preconditions;

import org.eclipse.collections.api.tuple.primitive.IntIntPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

class SplitFinder {
    private static final Logger LOG = LoggerFactory.getLogger(SplitFinder.class);

    private final int subtaskId;
    private final int numSubtasks;
    private final int[] numFeatureBins;
    private final HistogramFeatureSplitter[] splitters;
    private final int maxDepth;
    private final int maxNumLeaves;

    public SplitFinder(LocalState.Statics statics) {
        subtaskId = statics.subtaskId;
        numSubtasks = statics.numSubtasks;

        numFeatureBins = statics.numFeatureBins;
        FeatureMeta[] featureMetas = statics.featureMetas;
        splitters = new HistogramFeatureSplitter[statics.numFeatures];
        for (int i = 0; i < statics.numFeatures; ++i) {
            splitters[i] =
                    FeatureMeta.Type.CATEGORICAL == featureMetas[i].type
                            ? new CategoricalFeatureSplitter(i, featureMetas[i], statics.params)
                            : new ContinuousFeatureSplitter(i, featureMetas[i], statics.params);
        }
        maxDepth = statics.params.maxDepth;
        maxNumLeaves = statics.params.maxNumLeaves;
    }

    public Splits calc(
            List<LearningNode> layer,
            List<IntIntPair> nodeFeaturePairs,
            List<LearningNode> leaves,
            Histogram histogram) {
        LOG.info("subtaskId: {}, {} start", subtaskId, SplitFinder.class.getSimpleName());

        Distributor distributor =
                new Distributor.EvenDistributor(numSubtasks, nodeFeaturePairs.size());
        long start = distributor.start(subtaskId);
        long cnt = distributor.count(subtaskId);

        Split[] nodesBestSplits = new Split[layer.size()];
        int binOffset = 0;
        for (long i = start; i < start + cnt; i += 1) {
            IntIntPair nodeFeaturePair = nodeFeaturePairs.get((int) i);
            int nodeId = nodeFeaturePair.getOne();
            int featureId = nodeFeaturePair.getTwo();
            LearningNode node = layer.get(nodeId);

            Preconditions.checkState(node.depth < maxDepth || leaves.size() + 2 <= maxNumLeaves);
            splitters[featureId].reset(
                    histogram.hists, new Slice(binOffset, binOffset + numFeatureBins[featureId]));
            Split bestSplit = splitters[featureId].bestSplit();
            if (null == nodesBestSplits[nodeId]
                    || (bestSplit.gain > nodesBestSplits[nodeId].gain)) {
                nodesBestSplits[nodeId] = bestSplit;
            }
            binOffset += numFeatureBins[featureId];
        }

        LOG.info("subtaskId: {}, {} end", subtaskId, SplitFinder.class.getSimpleName());
        return new Splits(subtaskId, nodesBestSplits);
    }
}
