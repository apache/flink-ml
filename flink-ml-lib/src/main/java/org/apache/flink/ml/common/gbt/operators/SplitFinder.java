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

import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.splitter.CategoricalFeatureSplitter;
import org.apache.flink.ml.common.gbt.splitter.ContinuousFeatureSplitter;
import org.apache.flink.ml.common.gbt.splitter.HistogramFeatureSplitter;
import org.apache.flink.ml.util.Distributor;
import org.apache.flink.util.Preconditions;

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

    public SplitFinder(TrainContext trainContext) {
        subtaskId = trainContext.subtaskId;
        numSubtasks = trainContext.numSubtasks;

        numFeatureBins = trainContext.numFeatureBins;
        FeatureMeta[] featureMetas = trainContext.featureMetas;
        int numFeatures = trainContext.numFeatures;
        splitters = new HistogramFeatureSplitter[numFeatures + 1];
        for (int i = 0; i < numFeatures; ++i) {
            splitters[i] =
                    FeatureMeta.Type.CATEGORICAL == featureMetas[i].type
                            ? new CategoricalFeatureSplitter(
                                    i, featureMetas[i], trainContext.strategy)
                            : new ContinuousFeatureSplitter(
                                    i, featureMetas[i], trainContext.strategy);
        }
        // Adds an addition splitter to obtain the prediction of the node.
        splitters[numFeatures] =
                new ContinuousFeatureSplitter(
                        numFeatures,
                        new FeatureMeta.ContinuousFeatureMeta("SPECIAL", 0, new double[0]),
                        trainContext.strategy);
        maxDepth = trainContext.strategy.maxDepth;
        maxNumLeaves = trainContext.strategy.maxNumLeaves;
    }

    public Splits calc(
            List<LearningNode> layer, int[] nodeFeaturePairs, int numLeaves, Histogram histogram) {
        LOG.info("subtaskId: {}, {} start", subtaskId, SplitFinder.class.getSimpleName());

        Distributor distributor =
                new Distributor.EvenDistributor(numSubtasks, nodeFeaturePairs.length / 2);
        int start = (int) distributor.start(subtaskId);
        int cnt = (int) distributor.count(subtaskId);

        Split[] nodesBestSplits = new Split[layer.size()];
        int binOffset = 0;
        for (int i = start; i < start + cnt; i += 1) {
            int nodeId = nodeFeaturePairs[2 * i];
            int featureId = nodeFeaturePairs[2 * i + 1];
            LearningNode node = layer.get(nodeId);

            Preconditions.checkState(node.depth < maxDepth || numLeaves + 2 <= maxNumLeaves);
            Preconditions.checkState(histogram.slice.start == 0);
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
