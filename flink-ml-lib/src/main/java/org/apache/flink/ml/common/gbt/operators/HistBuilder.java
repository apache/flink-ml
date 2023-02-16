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
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.defs.TrainContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static org.apache.flink.ml.common.gbt.DataUtils.BIN_SIZE;

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

    public HistBuilder(TrainContext trainContext) {
        subtaskId = trainContext.subtaskId;
        numSubtasks = trainContext.numSubtasks;

        numFeatureBins = trainContext.numFeatureBins;
        featureMetas = trainContext.featureMetas;

        numBaggingFeatures = trainContext.numBaggingFeatures;
        featureRandomizer = trainContext.featureRandomizer;
        featureIndicesPool = IntStream.range(0, trainContext.numFeatures).toArray();

        isInputVector = trainContext.params.isInputVector;

        int maxNumNodes =
                Math.min(
                        ((int) Math.pow(2, trainContext.params.maxDepth - 1)),
                        trainContext.params.maxNumLeaves);

        int maxFeatureBins = Arrays.stream(numFeatureBins).max().orElse(0);
        int totalNumFeatureBins = Arrays.stream(numFeatureBins).sum();
        int maxNumBins =
                maxNumNodes * Math.min(maxFeatureBins * numBaggingFeatures, totalNumFeatureBins);
        hists = new double[maxNumBins * BIN_SIZE];
    }

    /**
     * Calculate histograms for all (nodeId, featureId) pairs. The results are written to `hists`,
     * so `hists` must be large enough to store values.
     */
    private static void calcNodeFeaturePairHists(
            List<LearningNode> layer,
            int[][] nodeToFeatures,
            FeatureMeta[] featureMetas,
            int[] numFeatureBins,
            boolean isInputVector,
            int[] indices,
            BinnedInstance[] instances,
            PredGradHess[] pgh,
            double[] hists) {

        int numNodes = layer.size();
        int numFeatures = featureMetas.length;

        int[][] nodeToBinOffsets = new int[numNodes][];
        int binOffset = 0;
        for (int k = 0; k < numNodes; k += 1) {
            int[] features = nodeToFeatures[k];
            nodeToBinOffsets[k] = new int[features.length];
            for (int i = 0; i < features.length; i += 1) {
                nodeToBinOffsets[k][i] = binOffset;
                binOffset += numFeatureBins[features[i]];
            }
        }

        int[] featureDefaultVal = new int[numFeatures];
        for (int i = 0; i < numFeatures; i += 1) {
            FeatureMeta d = featureMetas[i];
            featureDefaultVal[i] =
                    isInputVector && d instanceof FeatureMeta.ContinuousFeatureMeta
                            ? ((FeatureMeta.ContinuousFeatureMeta) d).zeroBin
                            : d.missingBin;
        }

        int[] featureOffset = new int[numFeatures];
        for (int k = 0; k < numNodes; k += 1) {
            int[] features = nodeToFeatures[k];
            int[] binOffsets = nodeToBinOffsets[k];
            LearningNode node = layer.get(k);

            BitSet featureValid = new BitSet(numFeatures);
            for (int i = 0; i < features.length; i += 1) {
                featureValid.set(features[i]);
                featureOffset[features[i]] = binOffsets[i];
            }

            double[] totalHists = new double[4];
            for (int i = node.slice.start; i < node.slice.end; i += 1) {
                int instanceId = indices[i];
                BinnedInstance binnedInstance = instances[instanceId];
                double weight = binnedInstance.weight;
                double gradient = pgh[instanceId].gradient;
                double hessian = pgh[instanceId].hessian;

                totalHists[0] += gradient;
                totalHists[1] += hessian;
                totalHists[2] += weight;
                totalHists[3] += 1.;

                if (null == binnedInstance.featureIds) {
                    for (int j = 0; j < binnedInstance.featureValues.length; j += 1) {
                        if (!featureValid.get(j)) {
                            continue;
                        }
                        int val = binnedInstance.featureValues[j];
                        int offset = featureOffset[j];
                        int index = (offset + val) * BIN_SIZE;
                        hists[index] += gradient;
                        hists[index + 1] += hessian;
                        hists[index + 2] += weight;
                        hists[index + 3] += 1.;
                    }
                } else {
                    for (int j = 0; j < binnedInstance.featureIds.length; j += 1) {
                        int featureId = binnedInstance.featureIds[j];
                        if (!featureValid.get(featureId)) {
                            continue;
                        }
                        int val = binnedInstance.featureValues[j];
                        int offset = featureOffset[featureId];
                        int index = (offset + val) * BIN_SIZE;
                        hists[index] += gradient;
                        hists[index + 1] += hessian;
                        hists[index + 2] += weight;
                        hists[index + 3] += 1.;
                    }
                }
            }

            for (int featureId : features) {
                int defaultVal = featureDefaultVal[featureId];
                int defaultValIndex = (featureOffset[featureId] + defaultVal) * BIN_SIZE;
                hists[defaultValIndex] = totalHists[0];
                hists[defaultValIndex + 1] = totalHists[1];
                hists[defaultValIndex + 2] = totalHists[2];
                hists[defaultValIndex + 3] = totalHists[3];

                for (int i = 0; i < numFeatureBins[featureId]; i += 1) {
                    if (i != defaultVal) {
                        int index = (featureOffset[featureId] + i) * BIN_SIZE;
                        hists[defaultValIndex] -= hists[index];
                        hists[defaultValIndex + 1] -= hists[index + 1];
                        hists[defaultValIndex + 2] -= hists[index + 2];
                        hists[defaultValIndex + 3] -= hists[index + 3];
                    }
                }
            }
        }
    }

    /**
     * Calculates elements counts of histogram distributed to each downstream subtask. The elements
     * counts is bin counts multiplied by STEP. The minimum unit to be distributed is (nodeId,
     * featureId), i.e., all bins belonging to the same (nodeId, featureId) pair must go to one
     * subtask.
     */
    private static int[] calcRecvCounts(
            int numSubtasks, int[] nodeFeaturePairs, int[] numFeatureBins) {
        int[] recvcnts = new int[numSubtasks];
        Distributor.EvenDistributor distributor =
                new Distributor.EvenDistributor(numSubtasks, nodeFeaturePairs.length / 2);
        for (int k = 0; k < numSubtasks; k += 1) {
            int pairStart = (int) distributor.start(k);
            int pairCnt = (int) distributor.count(k);
            for (int i = pairStart; i < pairStart + pairCnt; i += 1) {
                int featureId = nodeFeaturePairs[2 * i + 1];
                recvcnts[k] += numFeatureBins[featureId] * BIN_SIZE;
            }
        }
        return recvcnts;
    }

    /** Calculate local histograms for nodes in current layer of tree. */
    Histogram build(
            List<LearningNode> layer,
            int[] indices,
            BinnedInstance[] instances,
            PredGradHess[] pgh,
            Consumer<int[]> nodeFeaturePairsSetter) {
        LOG.info("subtaskId: {}, {} start", subtaskId, HistBuilder.class.getSimpleName());
        int numNodes = layer.size();

        // Generates (nodeId, featureId) pairs that are required to build histograms.
        int[][] nodeToFeatures = new int[numNodes][];
        int[] nodeFeaturePairs = new int[numNodes * numBaggingFeatures * 2];
        int p = 0;
        for (int k = 0; k < numNodes; k += 1) {
            nodeToFeatures[k] =
                    DataUtils.sample(featureIndicesPool, numBaggingFeatures, featureRandomizer);
            Arrays.sort(nodeToFeatures[k]);
            for (int featureId : nodeToFeatures[k]) {
                nodeFeaturePairs[p++] = k;
                nodeFeaturePairs[p++] = featureId;
            }
        }
        nodeFeaturePairsSetter.accept(nodeFeaturePairs);

        Arrays.fill(hists, 0);
        // Calculates histograms for (nodeId, featureId) pairs.
        calcNodeFeaturePairHists(
                layer,
                nodeToFeatures,
                featureMetas,
                numFeatureBins,
                isInputVector,
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
