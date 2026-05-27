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
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.lossfunc.LossFunc;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.function.Consumer;

class InstanceUpdater {
    private static final Logger LOG = LoggerFactory.getLogger(InstanceUpdater.class);

    private final int subtaskId;
    private final LossFunc loss;
    private final double stepSize;
    private final double prior;

    public InstanceUpdater(TrainContext trainContext) {
        subtaskId = trainContext.subtaskId;
        loss = trainContext.loss;
        stepSize = trainContext.strategy.stepSize;
        prior = trainContext.prior;
    }

    public void update(
            double[] pgh,
            List<LearningNode> leaves,
            int[] indices,
            BinnedInstance[] instances,
            Consumer<double[]> pghSetter,
            List<Node> treeNodes) {
        LOG.info("subtaskId: {}, {} start", subtaskId, InstanceUpdater.class.getSimpleName());
        long start = System.currentTimeMillis();
        if (pgh.length == 0) {
            pgh = new double[instances.length * 3];
            for (int i = 0; i < instances.length; i += 1) {
                double label = instances[i].label;
                pgh[3 * i] = prior;
                pgh[3 * i + 1] = loss.gradient(prior, label);
                pgh[3 * i + 2] = loss.hessian(prior, label);
            }
        }

        for (LearningNode nodeInfo : leaves) {
            Split split = treeNodes.get(nodeInfo.nodeIndex).split;
            double pred = split.prediction * stepSize;
            for (int i = nodeInfo.slice.start; i < nodeInfo.slice.end; ++i) {
                int instanceId = indices[i];
                updatePgh(instanceId, pred, instances[instanceId].label, pgh);
            }
            for (int i = nodeInfo.oob.start; i < nodeInfo.oob.end; ++i) {
                int instanceId = indices[i];
                updatePgh(instanceId, pred, instances[instanceId].label, pgh);
            }
        }
        pghSetter.accept(pgh);
        LOG.info("subtaskId: {}, {} end", subtaskId, InstanceUpdater.class.getSimpleName());
        LOG.info(
                "subtaskId: {}, elapsed time for updating instances: {} ms",
                subtaskId,
                System.currentTimeMillis() - start);
    }

    private void updatePgh(int instanceId, double pred, double label, double[] pgh) {
        pgh[instanceId * 3] += pred;
        pgh[instanceId * 3 + 1] = loss.gradient(pgh[instanceId * 3], label);
        pgh[instanceId * 3 + 2] = loss.hessian(pgh[instanceId * 3], label);
    }
}
