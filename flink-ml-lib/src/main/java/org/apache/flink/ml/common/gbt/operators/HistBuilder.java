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

import org.apache.flink.ml.common.gbt.DataUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.Distributor;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;

import org.eclipse.collections.api.tuple.primitive.IntIntPair;
import org.eclipse.collections.impl.tuple.primitive.PrimitiveTuples;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

class HistBuilder {
    private static final Logger LOG = LoggerFactory.getLogger(HistBuilder.class);

    private final int subtaskId;
    private final int numSubtasks;

    private final int[] numFeatureBins;
    private final FeatureMeta[] featureMetas;

    private final int numBaggingFeatures;
    private final Random featureRandomizer;
    private final int[] featureIndicesPool;

    private final boolean isInputVector;

    private final double[] hists;

    public HistBuilder(LocalState.Statics statics) {
        subtaskId = statics.subtaskId;
        numSubtasks = statics.numSubtasks;

        numFeatureBins = statics.numFeatureBins;
        featureMetas = statics.featureMetas;

        numBaggingFeatures = statics.numBaggingFeatures;
        featureRandomizer = statics.featureRandomizer;
        featureIndicesPool = IntStream.range(0, statics.numFeatures).toArray();

        isInputVector = statics.params.isInputVector;

        int maxNumNodes =
                Math.min(
                        ((int) Math.pow(2, statics.params.maxDepth - 1)),
                        statics.params.maxNumLeaves);

        int maxFeatureBins = Arrays.stream(numFeatureBins).max().orElse(0);
        int totalNumFeatureBins = Arrays.stream(numFeatureBins).sum();
        int maxNumBins =
                maxNumNodes * Math.min(maxFeatureBins * numBaggingFeatures, totalNumFeatureBins);
        hists = new double[maxNumBins * DataUtils.BIN_SIZE];
    }

    /**
     * Calculate histograms for all (nodeId, featureId) pairs. The results are written to `hists`,
     * so `hists` must be large enough to store values.
     */
    private static void calcNodeFeaturePairHists(
            List<LearningNode> layer,
            List<IntIntPair> nodeFeaturePairs,
            FeatureMeta[] featureMetas,
            boolean isInputVector,
            int[] numFeatureBins,
            int[] indices,
            BinnedInstance[] instances,
            PredGradHess[] pgh,
            double[] hists) {
        Arrays.fill(hists, 0.);
        int binOffset = 0;
        for (IntIntPair nodeFeaturePair : nodeFeaturePairs) {
            int nodeId = nodeFeaturePair.getOne();
            int featureId = nodeFeaturePair.getTwo();
            FeatureMeta featureMeta = featureMetas[featureId];

            int defaultValue = featureMeta.missingBin;
            // When isInputVector is true, values of unseen features are treated as 0s.
            if (isInputVector && featureMeta instanceof FeatureMeta.ContinuousFeatureMeta) {
                defaultValue = ((FeatureMeta.ContinuousFeatureMeta) featureMeta).zeroBin;
            }

            LearningNode node = layer.get(nodeId);
            for (int i = node.slice.start; i < node.slice.end; i += 1) {
                int instanceId = indices[i];
                BinnedInstance binnedInstance = instances[instanceId];
                double gradient = pgh[instanceId].gradient;
                double hessian = pgh[instanceId].hessian;

                int val = binnedInstance.features.getIfAbsent(featureId, defaultValue);
                int startIndex = (binOffset + val) * DataUtils.BIN_SIZE;
                hists[startIndex] += gradient;
                hists[startIndex + 1] += hessian;
                hists[startIndex + 2] += binnedInstance.weight;
                hists[startIndex + 3] += 1.;
            }
            binOffset += numFeatureBins[featureId];
        }
    }

    /**
     * Calculates elements counts of histogram distributed to each downstream subtask. The elements
     * counts is bin counts multiplied by STEP. The minimum unit to be distributed is (nodeId,
     * featureId), i.e., all bins belonging to the same (nodeId, featureId) pair must go to one
     * subtask.
     */
    private static int[] calcRecvCounts(
            int numSubtasks, List<IntIntPair> nodeFeaturePairs, int[] numFeatureBins) {
        int[] recvcnts = new int[numSubtasks];
        Distributor.EvenDistributor distributor =
                new Distributor.EvenDistributor(numSubtasks, nodeFeaturePairs.size());
        for (int k = 0; k < numSubtasks; k += 1) {
            int pairStart = (int) distributor.start(k);
            int pairCnt = (int) distributor.count(k);
            for (int i = pairStart; i < pairStart + pairCnt; i += 1) {
                int featureId = nodeFeaturePairs.get(i).getTwo();
                recvcnts[k] += numFeatureBins[featureId] * DataUtils.BIN_SIZE;
            }
        }
        return recvcnts;
    }

    /** Calculate local histograms for nodes in current layer of tree. */
    public Histogram build(
            List<LearningNode> layer,
            List<IntIntPair> nodeFeaturePairs,
            int[] indices,
            BinnedInstance[] instances,
            PredGradHess[] pgh) {
        LOG.info("subtaskId: {}, {} start", subtaskId, HistBuilder.class.getSimpleName());

        // Generates (nodeId, featureId) pairs that are required to build histograms.
        nodeFeaturePairs.clear();
        for (int k = 0; k < layer.size(); k += 1) {
            int[] sampledFeatures =
                    DataUtils.sample(featureIndicesPool, numBaggingFeatures, featureRandomizer);
            for (int featureId : sampledFeatures) {
                nodeFeaturePairs.add(PrimitiveTuples.pair(k, featureId));
            }
        }

        // Calculates histograms for (nodeId, featureId) pairs.
        calcNodeFeaturePairHists(
                layer,
                nodeFeaturePairs,
                featureMetas,
                isInputVector,
                numFeatureBins,
                indices,
                instances,
                pgh,
                hists);

        // Calculates number of elements received by each downstream subtask.
        int[] recvcnts = calcRecvCounts(numSubtasks, nodeFeaturePairs, numFeatureBins);

        LOG.info("subtaskId: {}, {} end", this.subtaskId, HistBuilder.class.getSimpleName());
        return new Histogram(this.subtaskId, hists, recvcnts);
    }
}
