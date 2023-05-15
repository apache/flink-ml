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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.sharedobjects.AbstractSharedObjectsOneInputStreamOperator;
import org.apache.flink.ml.common.sharedobjects.ReadRequest;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.ALL_TREES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.CURRENT_TREE_NODES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.INSTANCES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.LAYER;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.LEAVES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.NEED_INIT_TREE;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.PREDS_GRADS_HESSIANS;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.ROOT_LEARNING_NODE;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.SHUFFLED_INDICES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.SWAPPED_INDICES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.TRAIN_CONTEXT;

/**
 * Post-process after global splits obtained, including split instances to left or child nodes, and
 * update instances scores after a tree is complete.
 */
public class PostSplitsOperator
        extends AbstractSharedObjectsOneInputStreamOperator<Tuple2<Integer, Split>, Integer>
        implements IterationListener<Integer> {

    private static final String NODE_SPLITTER_STATE_NAME = "node_splitter";
    private static final String INSTANCE_UPDATER_STATE_NAME = "instance_updater";

    private static final Logger LOG = LoggerFactory.getLogger(PostSplitsOperator.class);

    // States of local data.
    private transient Split[] nodeSplits;
    private transient ListStateWithCache<NodeSplitter> nodeSplitterState;
    private transient NodeSplitter nodeSplitter;
    private transient ListStateWithCache<InstanceUpdater> instanceUpdaterState;
    private transient InstanceUpdater instanceUpdater;

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);

        nodeSplitterState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(NodeSplitter.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        nodeSplitter =
                OperatorStateUtils.getUniqueElement(nodeSplitterState, NODE_SPLITTER_STATE_NAME)
                        .orElse(null);
        instanceUpdaterState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(InstanceUpdater.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        instanceUpdater =
                OperatorStateUtils.getUniqueElement(
                                instanceUpdaterState, INSTANCE_UPDATER_STATE_NAME)
                        .orElse(null);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        nodeSplitterState.snapshotState(context);
        instanceUpdaterState.snapshotState(context);
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context c, Collector<Integer> collector) throws Exception {
        if (0 == epochWatermark) {
            TrainContext trainContext = context.read(TRAIN_CONTEXT.sameStep());
            nodeSplitter = new NodeSplitter(trainContext);
            nodeSplitterState.update(Collections.singletonList(nodeSplitter));
            instanceUpdater = new InstanceUpdater(trainContext);
            instanceUpdaterState.update(Collections.singletonList(instanceUpdater));
        }

        int[] indices = new int[0];
        if (epochWatermark > 0) {
            indices = context.read(SWAPPED_INDICES.prevStep());
        }
        if (0 == indices.length) {
            indices = context.read(SHUFFLED_INDICES.sameStep()).clone();
        }

        BinnedInstance[] instances = context.read(INSTANCES.sameStep());
        List<LearningNode> leaves = context.read(LEAVES.prevStep());
        List<LearningNode> layer = context.read(LAYER.prevStep());
        List<Node> currentTreeNodes;
        if (layer.isEmpty()) {
            layer = Collections.singletonList(context.read(ROOT_LEARNING_NODE.sameStep()));
            currentTreeNodes = new ArrayList<>();
            currentTreeNodes.add(new Node());
        } else {
            currentTreeNodes = context.read(CURRENT_TREE_NODES.prevStep());
        }

        List<LearningNode> nextLayer =
                nodeSplitter.split(currentTreeNodes, layer, leaves, nodeSplits, indices, instances);
        nodeSplits = null;
        context.write(LEAVES, leaves);
        context.write(LAYER, nextLayer);
        context.write(CURRENT_TREE_NODES, currentTreeNodes);

        if (nextLayer.isEmpty()) {
            // Current tree is finished.
            context.write(NEED_INIT_TREE, true);
            instanceUpdater.update(
                    context.read(PREDS_GRADS_HESSIANS.prevStep()),
                    leaves,
                    indices,
                    instances,
                    d -> context.write(PREDS_GRADS_HESSIANS, d),
                    currentTreeNodes);
            leaves.clear();
            List<List<Node>> allTrees = context.read(ALL_TREES.prevStep());
            allTrees.add(currentTreeNodes);

            context.write(LEAVES, new ArrayList<>());
            context.write(SWAPPED_INDICES, new int[0]);
            context.write(ALL_TREES, allTrees);
            LOG.info("finalize {}-th tree", allTrees.size());
        } else {
            context.write(SWAPPED_INDICES, indices);
            context.write(NEED_INIT_TREE, false);

            context.renew(PREDS_GRADS_HESSIANS);
            context.renew(ALL_TREES);
        }
    }

    @Override
    public void onIterationTerminated(Context c, Collector<Integer> collector) {
        context.write(PREDS_GRADS_HESSIANS, new double[0]);
        context.write(SWAPPED_INDICES, new int[0]);
        context.write(LEAVES, Collections.emptyList());
        context.write(LAYER, Collections.emptyList());
        context.write(CURRENT_TREE_NODES, Collections.emptyList());
    }

    @Override
    public void processElement(StreamRecord<Tuple2<Integer, Split>> element) throws Exception {
        if (null == nodeSplits) {
            List<LearningNode> layer = context.read(LAYER.sameStep());
            int numNodes = (layer.isEmpty()) ? 1 : layer.size();
            nodeSplits = new Split[numNodes];
        }
        Tuple2<Integer, Split> value = element.getValue();
        int nodeId = value.f0;
        Split split = value.f1;
        LOG.debug("Received split for node {}", nodeId);
        nodeSplits[nodeId] = split;
    }

    @Override
    public List<ReadRequest<?>> readRequestsInProcessElement() {
        return Collections.singletonList(LAYER.sameStep());
    }

    @Override
    public void close() throws Exception {
        nodeSplitterState.clear();
        instanceUpdaterState.clear();
        super.close();
    }
}
