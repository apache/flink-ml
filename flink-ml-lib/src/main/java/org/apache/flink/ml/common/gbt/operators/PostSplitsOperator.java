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
import org.apache.flink.ml.common.sharedobjects.SharedObjectsContext;
import org.apache.flink.ml.common.sharedobjects.SharedObjectsStreamOperator;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Post-process after global splits obtained, including split instances to left or child nodes, and
 * update instances scores after a tree is complete.
 */
public class PostSplitsOperator extends AbstractStreamOperator<Integer>
        implements OneInputStreamOperator<Tuple2<Integer, Split>, Integer>,
                IterationListener<Integer>,
                SharedObjectsStreamOperator {

    private static final String NODE_SPLITTER_STATE_NAME = "node_splitter";
    private static final String INSTANCE_UPDATER_STATE_NAME = "instance_updater";

    private static final Logger LOG = LoggerFactory.getLogger(PostSplitsOperator.class);

    private final String sharedObjectsAccessorID;

    // States of local data.
    private transient Split[] nodeSplits;
    private transient ListStateWithCache<NodeSplitter> nodeSplitterState;
    private transient NodeSplitter nodeSplitter;
    private transient ListStateWithCache<InstanceUpdater> instanceUpdaterState;
    private transient InstanceUpdater instanceUpdater;
    private transient SharedObjectsContext sharedObjectsContext;

    public PostSplitsOperator() {
        sharedObjectsAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
    }

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
            int epochWatermark, Context context, Collector<Integer> collector) throws Exception {
        if (0 == epochWatermark) {
            sharedObjectsContext.invoke(
                    (getter, setter) -> {
                        TrainContext trainContext =
                                getter.get(SharedObjectsConstants.TRAIN_CONTEXT);
                        nodeSplitter = new NodeSplitter(trainContext);
                        nodeSplitterState.update(Collections.singletonList(nodeSplitter));
                        instanceUpdater = new InstanceUpdater(trainContext);
                        instanceUpdaterState.update(Collections.singletonList(instanceUpdater));
                    });
        }

        sharedObjectsContext.invoke(
                (getter, setter) -> {
                    int[] indices = getter.get(SharedObjectsConstants.SWAPPED_INDICES);
                    if (0 == indices.length) {
                        indices = getter.get(SharedObjectsConstants.SHUFFLED_INDICES).clone();
                    }

                    BinnedInstance[] instances = getter.get(SharedObjectsConstants.INSTANCES);
                    List<LearningNode> leaves = getter.get(SharedObjectsConstants.LEAVES);
                    List<LearningNode> layer = getter.get(SharedObjectsConstants.LAYER);
                    List<Node> currentTreeNodes;
                    if (layer.size() == 0) {
                        layer =
                                Collections.singletonList(
                                        getter.get(SharedObjectsConstants.ROOT_LEARNING_NODE));
                        currentTreeNodes = new ArrayList<>();
                        currentTreeNodes.add(new Node());
                    } else {
                        currentTreeNodes = getter.get(SharedObjectsConstants.CURRENT_TREE_NODES);
                    }

                    List<LearningNode> nextLayer =
                            nodeSplitter.split(
                                    currentTreeNodes,
                                    layer,
                                    leaves,
                                    nodeSplits,
                                    indices,
                                    instances);
                    nodeSplits = null;
                    setter.set(SharedObjectsConstants.LEAVES, leaves);
                    setter.set(SharedObjectsConstants.LAYER, nextLayer);
                    setter.set(SharedObjectsConstants.CURRENT_TREE_NODES, currentTreeNodes);

                    if (nextLayer.isEmpty()) {
                        // Current tree is finished.
                        setter.set(SharedObjectsConstants.NEED_INIT_TREE, true);
                        instanceUpdater.update(
                                getter.get(SharedObjectsConstants.PREDS_GRADS_HESSIANS),
                                leaves,
                                indices,
                                instances,
                                d -> setter.set(SharedObjectsConstants.PREDS_GRADS_HESSIANS, d),
                                currentTreeNodes);
                        leaves.clear();
                        List<List<Node>> allTrees = getter.get(SharedObjectsConstants.ALL_TREES);
                        allTrees.add(currentTreeNodes);

                        setter.set(SharedObjectsConstants.LEAVES, new ArrayList<>());
                        setter.set(SharedObjectsConstants.SWAPPED_INDICES, new int[0]);
                        setter.set(SharedObjectsConstants.ALL_TREES, allTrees);
                        LOG.info("finalize {}-th tree", allTrees.size());
                    } else {
                        setter.set(SharedObjectsConstants.SWAPPED_INDICES, indices);
                        setter.set(SharedObjectsConstants.NEED_INIT_TREE, false);
                    }
                });
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Integer> collector)
            throws Exception {
        sharedObjectsContext.invoke(
                (getter, setter) -> {
                    setter.set(SharedObjectsConstants.PREDS_GRADS_HESSIANS, new double[0]);
                    setter.set(SharedObjectsConstants.SWAPPED_INDICES, new int[0]);
                    setter.set(SharedObjectsConstants.LEAVES, Collections.emptyList());
                    setter.set(SharedObjectsConstants.LAYER, Collections.emptyList());
                    setter.set(SharedObjectsConstants.CURRENT_TREE_NODES, Collections.emptyList());
                });
    }

    @Override
    public void processElement(StreamRecord<Tuple2<Integer, Split>> element) throws Exception {
        if (null == nodeSplits) {
            sharedObjectsContext.invoke(
                    (getter, setter) -> {
                        List<LearningNode> layer = getter.get(SharedObjectsConstants.LAYER);
                        int numNodes = (layer.size() == 0) ? 1 : layer.size();
                        nodeSplits = new Split[numNodes];
                    });
        }
        Tuple2<Integer, Split> value = element.getValue();
        int nodeId = value.f0;
        Split split = value.f1;
        LOG.debug("Received split for node {}", nodeId);
        nodeSplits[nodeId] = split;
    }

    @Override
    public void close() throws Exception {
        nodeSplitterState.clear();
        instanceUpdaterState.clear();
        super.close();
    }

    @Override
    public void onSharedObjectsContextSet(SharedObjectsContext context) {
        sharedObjectsContext = context;
    }

    @Override
    public String getSharedObjectsAccessorID() {
        return sharedObjectsAccessorID;
    }
}
