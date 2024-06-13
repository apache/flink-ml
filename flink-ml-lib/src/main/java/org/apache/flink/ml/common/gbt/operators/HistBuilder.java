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

import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.gbt.DataUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
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

    private final int numFeatures;
    private final int[] numFeatureBins;
    private final FeatureMeta[] featureMetas;

    private final int numBaggingFeatures;
    private final Random featureRandomizer;
    private final int[] featureIndicesPool;

    private final boolean isInputVector;
    private final int maxDepth;

    public HistBuilder(TrainContext trainContext) {
        subtaskId = trainContext.subtaskId;

        numFeatures = trainContext.numFeatures;
        numFeatureBins = trainContext.numFeatureBins;
        featureMetas = trainContext.featureMetas;

        numBaggingFeatures = trainContext.numBaggingFeatures;
        featureRandomizer = trainContext.featureRandomizer;
        featureIndicesPool = IntStream.range(0, trainContext.numFeatures).toArray();

        isInputVector = trainContext.strategy.isInputVector;
        maxDepth = trainContext.strategy.maxDepth;
    }

    /** Calculate local histograms for nodes in current layer of tree. */
    void build(
            List<LearningNode> layer,
            int[] indices,
            BinnedInstance[] instances,
            double[] pgh,
            Consumer<int[]> nodeFeaturePairsSetter,
            Collector<Tuple3<Integer, Integer, Histogram>> out) {
        LOG.info("subtaskId: {}, {} start", subtaskId, HistBuilder.class.getSimpleName());
        int numNodes = layer.size();

        // Generates (nodeId, featureId) pairs that are required to build histograms.
        int[][] nodeToFeatures = new int[numNodes][];
        IntArrayList nodeFeaturePairs = new IntArrayList(numNodes * numBaggingFeatures * 2);
        for (int k = 0; k < numNodes; k += 1) {
            LearningNode node = layer.get(k);
            if (node.depth == maxDepth) {
                // Ignores the results, just to consume the randomizer.
                DataUtils.sample(featureIndicesPool, numBaggingFeatures, featureRandomizer);
                // No need to calculate histograms for features, only sum of gradients and hessians
                // are needed. Uses `numFeatures` to indicate this special "feature".
                nodeToFeatures[k] = new int[] {numFeatures};
            } else {
                nodeToFeatures[k] =
                        DataUtils.sample(featureIndicesPool, numBaggingFeatures, featureRandomizer);
                Arrays.sort(nodeToFeatures[k]);
            }
            for (int featureId : nodeToFeatures[k]) {
                nodeFeaturePairs.add(k);
                nodeFeaturePairs.add(featureId);
            }
        }
        nodeFeaturePairsSetter.accept(nodeFeaturePairs.toArray());

        // Calculates histograms for (nodeId, featureId) pairs.
        HistBuilderImpl builderImpl =
                new HistBuilderImpl(
                        layer,
                        maxDepth,
                        numFeatures,
                        numFeatureBins,
                        nodeToFeatures,
                        indices,
                        instances,
                        pgh);
        builderImpl.init(isInputVector, featureMetas);
        builderImpl.calcHistsForPairs(subtaskId, out);

        LOG.info("subtaskId: {}, {} end", subtaskId, HistBuilder.class.getSimpleName());
    }

    static class HistBuilderImpl {
        private final List<LearningNode> layer;
        private final int maxDepth;
        private final int numFeatures;
        private final int[] numFeatureBins;
        private final int[][] nodeToFeatures;
        private final int[] indices;
        private final BinnedInstance[] instances;
        private final double[] pgh;

        private int[] featureDefaultVal;

        public HistBuilderImpl(
                List<LearningNode> layer,
                int maxDepth,
                int numFeatures,
                int[] numFeatureBins,
                int[][] nodeToFeatures,
                int[] indices,
                BinnedInstance[] instances,
                double[] pgh) {
            this.layer = layer;
            this.maxDepth = maxDepth;
            this.numFeatures = numFeatures;
            this.numFeatureBins = numFeatureBins;
            this.nodeToFeatures = nodeToFeatures;
            this.indices = indices;
            this.instances = instances;
            this.pgh = pgh;
            Preconditions.checkArgument(numFeatureBins.length == numFeatures + 1);
        }

        private static void calcHistsForDefaultBin(
                int defaultVal,
                int featureOffset,
                int numBins,
                double[] totalHists,
                double[] hists,
                int nodeOffset) {
            int defaultValIndex = (nodeOffset + featureOffset + defaultVal) * BIN_SIZE;
            hists[defaultValIndex] = totalHists[0];
            hists[defaultValIndex + 1] = totalHists[1];
            hists[defaultValIndex + 2] = totalHists[2];
            hists[defaultValIndex + 3] = totalHists[3];
            for (int i = 0; i < numBins; i += 1) {
                if (i != defaultVal) {
                    int index = (nodeOffset + featureOffset + i) * BIN_SIZE;
                    add(
                            hists,
                            nodeOffset + featureOffset,
                            defaultVal,
                            -hists[index],
                            -hists[index + 1],
                            -hists[index + 2],
                            -hists[index + 3]);
                }
            }
        }

        private static void add(
                double[] hists, int offset, int val, double d0, double d1, double d2, double d3) {
            int index = (offset + val) * BIN_SIZE;
            hists[index] += d0;
            hists[index + 1] += d1;
            hists[index + 2] += d2;
            hists[index + 3] += d3;
        }

        private void init(boolean isInputVector, FeatureMeta[] featureMetas) {
            featureDefaultVal = new int[numFeatures];
            for (int i = 0; i < numFeatures; i += 1) {
                FeatureMeta d = featureMetas[i];
                featureDefaultVal[i] =
                        isInputVector && d instanceof FeatureMeta.ContinuousFeatureMeta
                                ? ((FeatureMeta.ContinuousFeatureMeta) d).zeroBin
                                : d.missingBin;
            }
        }

        private void calcTotalHists(LearningNode node, double[] totalHists, int offset) {
            for (int i = node.slice.start; i < node.slice.end; i += 1) {
                int instanceId = indices[i];
                BinnedInstance binnedInstance = instances[instanceId];
                double weight = binnedInstance.weight;
                double gradient = pgh[3 * instanceId + 1];
                double hessian = pgh[3 * instanceId + 2];
                add(totalHists, offset, 0, gradient, hessian, weight, 1.);
            }
        }

        private void calcHistsForNonDefaultBins(
                LearningNode node,
                boolean allFeatureValid,
                BitSet featureValid,
                int[] featureOffset,
                double[] hists,
                int nodeOffset) {
            for (int i = node.slice.start; i < node.slice.end; i += 1) {
                int instanceId = indices[i];
                BinnedInstance binnedInstance = instances[instanceId];
                double weight = binnedInstance.weight;
                double gradient = pgh[3 * instanceId + 1];
                double hessian = pgh[3 * instanceId + 2];

                if (null == binnedInstance.featureIds) {
                    for (int j = 0; j < binnedInstance.featureValues.length; j += 1) {
                        if (allFeatureValid || featureValid.get(j)) {
                            add(
                                    hists,
                                    nodeOffset + featureOffset[j],
                                    binnedInstance.featureValues[j],
                                    gradient,
                                    hessian,
                                    weight,
                                    1.);
                        }
                    }
                } else {
                    for (int j = 0; j < binnedInstance.featureIds.length; j += 1) {
                        int featureId = binnedInstance.featureIds[j];
                        if (allFeatureValid || featureValid.get(featureId)) {
                            add(
                                    hists,
                                    nodeOffset + featureOffset[featureId],
                                    binnedInstance.featureValues[j],
                                    gradient,
                                    hessian,
                                    weight,
                                    1.);
                        }
                    }
                }
            }
        }

        private void calcHistsForSplitNode(
                LearningNode node,
                int[] features,
                int[] binOffsets,
                double[] hists,
                int nodeOffset) {
            double[] totalHists = new double[4];
            calcTotalHists(node, totalHists, 0);

            int[] featureOffsets = new int[numFeatures];
            BitSet featureValid = null;
            boolean allFeatureValid;
            if (numFeatures != features.length) {
                allFeatureValid = false;
                featureValid = new BitSet(numFeatures);
                for (int i = 0; i < features.length; i += 1) {
                    featureValid.set(features[i]);
                    featureOffsets[features[i]] = binOffsets[i];
                }
            } else {
                allFeatureValid = true;
                System.arraycopy(binOffsets, 0, featureOffsets, 0, numFeatures);
            }

            calcHistsForNonDefaultBins(
                    node, allFeatureValid, featureValid, featureOffsets, hists, nodeOffset);

            for (int featureId : features) {
                calcHistsForDefaultBin(
                        featureDefaultVal[featureId],
                        featureOffsets[featureId],
                        numFeatureBins[featureId],
                        totalHists,
                        hists,
                        nodeOffset);
            }
        }

        /** Calculate histograms for all (nodeId, featureId) pairs. */
        private void calcHistsForPairs(
                int subtaskId, Collector<Tuple3<Integer, Integer, Histogram>> out) {
            long start = System.currentTimeMillis();
            int numNodes = layer.size();
            int offset = 0;
            int pairBaseId = 0;
            for (int k = 0; k < numNodes; k += 1) {
                int[] features = nodeToFeatures[k];
                final int nodeOffset = offset;
                int[] binOffsets = new int[features.length];
                for (int i = 0; i < features.length; i += 1) {
                    binOffsets[i] = offset - nodeOffset;
                    offset += numFeatureBins[features[i]];
                }

                double[] nodeHists = new double[(offset - nodeOffset) * BIN_SIZE];
                long nodeStart = System.currentTimeMillis();
                LearningNode node = layer.get(k);
                if (node.depth != maxDepth) {
                    calcHistsForSplitNode(node, features, binOffsets, nodeHists, 0);
                } else {
                    calcTotalHists(node, nodeHists, 0);
                }
                LOG.info(
                        "subtaskId: {}, node {}, {} #instances, {} #features, {} ms",
                        subtaskId,
                        k,
                        node.slice.size(),
                        features.length,
                        System.currentTimeMillis() - nodeStart);

                int sliceStart = 0;
                for (int i = 0; i < features.length; i += 1) {
                    int sliceSize = numFeatureBins[features[i]] * BIN_SIZE;
                    int pairId = pairBaseId + i;
                    out.collect(
                            Tuple3.of(
                                    subtaskId,
                                    pairId,
                                    new Histogram(
                                            nodeHists,
                                            new Slice(sliceStart, sliceStart + sliceSize))));
                    sliceStart += sliceSize;
                }
                pairBaseId += features.length;
            }

            LOG.info(
                    "subtaskId: {}, elapsed time for calculating histograms: {} ms",
                    subtaskId,
                    System.currentTimeMillis() - start);
        }
    }
}
