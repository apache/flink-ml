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

import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.util.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

class NodeSplitter {
    private static final Logger LOG = LoggerFactory.getLogger(NodeSplitter.class);

    private final int subtaskId;
    private final FeatureMeta[] featureMetas;
    private final int maxLeaves;
    private final int maxDepth;

    public NodeSplitter(TrainContext trainContext) {
        subtaskId = trainContext.subtaskId;
        featureMetas = trainContext.featureMetas;
        maxLeaves = trainContext.strategy.maxNumLeaves;
        maxDepth = trainContext.strategy.maxDepth;
    }

    private int partitionInstances(
            Split split, Slice slice, int[] indices, BinnedInstance[] instances) {
        int lstart = slice.start;
        int lend = slice.end - 1;
        while (lstart <= lend) {
            while (lstart <= lend && split.shouldGoLeft(instances[indices[lstart]])) {
                lstart += 1;
            }
            while (lstart <= lend && !split.shouldGoLeft(instances[indices[lend]])) {
                lend -= 1;
            }
            if (lstart < lend) {
                int temp = indices[lstart];
                indices[lstart] = indices[lend];
                indices[lend] = temp;
            }
        }
        return lstart;
    }

    private void splitNode(
            Node treeNode,
            LearningNode nodeInfo,
            int[] indices,
            BinnedInstance[] instances,
            List<LearningNode> nextLayer,
            List<Node> treeNodes) {
        int mid = partitionInstances(treeNode.split, nodeInfo.slice, indices, instances);
        int oobMid = partitionInstances(treeNode.split, nodeInfo.oob, indices, instances);

        treeNode.left = treeNodes.size();
        treeNodes.add(new Node());
        treeNode.right = treeNodes.size();
        treeNodes.add(new Node());

        nextLayer.add(
                new LearningNode(
                        treeNode.left,
                        new Slice(nodeInfo.slice.start, mid),
                        new Slice(nodeInfo.oob.start, oobMid),
                        nodeInfo.depth + 1));
        nextLayer.add(
                new LearningNode(
                        treeNode.right,
                        new Slice(mid, nodeInfo.slice.end),
                        new Slice(oobMid, nodeInfo.oob.end),
                        nodeInfo.depth + 1));
    }

    public List<LearningNode> split(
            List<Node> treeNodes,
            List<LearningNode> layer,
            List<LearningNode> leaves,
            Split[] splits,
            int[] indices,
            BinnedInstance[] instances) {
        LOG.info("subtaskId: {}, {} start", subtaskId, NodeSplitter.class.getSimpleName());
        long start = System.currentTimeMillis();
        Preconditions.checkState(splits.length == layer.size());

        List<LearningNode> nextLayer = new ArrayList<>();

        // nodes in current layer or next layer are expected to generate at least 1 leaf.
        int numQueued = layer.size();
        for (int i = 0; i < layer.size(); i += 1) {
            LearningNode node = layer.get(i);
            Split split = splits[i];
            numQueued -= 1;
            Node treeNode = treeNodes.get(node.nodeIndex);
            treeNode.split = split;
            if (!split.isValid()
                    || treeNode.isLeaf
                    || (leaves.size() + numQueued + 2) > maxLeaves
                    || node.depth + 1 > maxDepth) {
                treeNode.isLeaf = true;
                leaves.add(node);
            } else {
                splitNode(treeNode, node, indices, instances, nextLayer, treeNodes);
                // Converts splits point from bin id to real feature value after splitting node.
                if (split instanceof Split.ContinuousSplit) {
                    Split.ContinuousSplit cs = (Split.ContinuousSplit) split;
                    FeatureMeta.ContinuousFeatureMeta featureMeta =
                            (FeatureMeta.ContinuousFeatureMeta) featureMetas[cs.featureId];
                    cs.threshold = featureMeta.binEdges[(int) cs.threshold + 1];
                }
                numQueued += 2;
            }
        }
        LOG.info("subtaskId: {}, {} end", subtaskId, NodeSplitter.class.getSimpleName());
        LOG.info(
                "subtaskId: {}, elapsed time for splitting nodes: {} ms",
                subtaskId,
                System.currentTimeMillis() - start);
        return nextLayer;
    }
}
