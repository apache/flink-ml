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
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
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
        stepSize = trainContext.params.stepSize;
        prior = trainContext.prior;
    }

    public void update(
            PredGradHess[] pgh,
            List<LearningNode> leaves,
            int[] indices,
            BinnedInstance[] instances,
            Consumer<PredGradHess[]> pghSetter,
            List<Node> treeNodes) {
        LOG.info("subtaskId: {}, {} start", subtaskId, InstanceUpdater.class.getSimpleName());
        if (pgh.length == 0) {
            pgh = new PredGradHess[instances.length];
            for (int i = 0; i < instances.length; i += 1) {
                double label = instances[i].label;
                pgh[i] =
                        new PredGradHess(
                                prior, loss.gradient(prior, label), loss.hessian(prior, label));
            }
        }

        for (LearningNode nodeInfo : leaves) {
            Split split = treeNodes.get(nodeInfo.nodeIndex).split;
            double pred = split.prediction * stepSize;
            for (int i = nodeInfo.slice.start; i < nodeInfo.slice.end; ++i) {
                int instanceId = indices[i];
                updatePgh(pred, instances[instanceId].label, pgh[instanceId]);
            }
            for (int i = nodeInfo.oob.start; i < nodeInfo.oob.end; ++i) {
                int instanceId = indices[i];
                updatePgh(pred, instances[instanceId].label, pgh[instanceId]);
            }
        }
        pghSetter.accept(pgh);
        LOG.info("subtaskId: {}, {} end", subtaskId, InstanceUpdater.class.getSimpleName());
    }

    private void updatePgh(double pred, double label, PredGradHess pgh) {
        pgh.pred += pred;
        pgh.gradient = loss.gradient(pgh.pred, label);
        pgh.hessian = loss.hessian(pgh.pred, label);
    }
}
