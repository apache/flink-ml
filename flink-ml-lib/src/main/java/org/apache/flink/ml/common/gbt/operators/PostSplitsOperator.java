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

import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.sharedstorage.SharedStorageContext;
import org.apache.flink.ml.common.sharedstorage.SharedStorageStreamOperator;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Post-process after global splits obtained, including split instances to left or child nodes, and
 * update instances scores after a tree is complete.
 */
public class PostSplitsOperator extends AbstractStreamOperator<Integer>
        implements OneInputStreamOperator<Splits, Integer>,
                IterationListener<Integer>,
                SharedStorageStreamOperator {

    private static final String SPLITS_STATE_NAME = "splits";
    private static final String NODE_SPLITTER_STATE_NAME = "node_splitter";
    private static final String INSTANCE_UPDATER_STATE_NAME = "instance_updater";

    private final String sharedStorageAccessorID;

    // States of local data.
    private transient ListStateWithCache<Splits> splitsState;
    private transient Splits splits;
    private transient ListStateWithCache<NodeSplitter> nodeSplitterState;
    private transient NodeSplitter nodeSplitter;
    private transient ListStateWithCache<InstanceUpdater> instanceUpdaterState;
    private transient InstanceUpdater instanceUpdater;
    private transient SharedStorageContext sharedStorageContext;

    public PostSplitsOperator() {
        sharedStorageAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);

        splitsState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(Splits.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        splits = OperatorStateUtils.getUniqueElement(splitsState, SPLITS_STATE_NAME).orElse(null);
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

        sharedStorageContext.initializeState(this, getRuntimeContext(), context);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        splitsState.snapshotState(context);
        nodeSplitterState.snapshotState(context);
        instanceUpdaterState.snapshotState(context);
        sharedStorageContext.snapshotState(context);
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Integer> collector) throws Exception {
        if (0 == epochWatermark) {
            sharedStorageContext.invoke(
                    (getter, setter) -> {
                        TrainContext trainContext =
                                getter.get(SharedStorageConstants.TRAIN_CONTEXT);
                        nodeSplitter = new NodeSplitter(trainContext);
                        nodeSplitterState.update(Collections.singletonList(nodeSplitter));
                        instanceUpdater = new InstanceUpdater(trainContext);
                        instanceUpdaterState.update(Collections.singletonList(instanceUpdater));
                    });
        }

        sharedStorageContext.invoke(
                (getter, setter) -> {
                    int[] indices = getter.get(SharedStorageConstants.SWAPPED_INDICES);
                    if (0 == indices.length) {
                        indices = getter.get(SharedStorageConstants.SHUFFLED_INDICES).clone();
                    }

                    BinnedInstance[] instances = getter.get(SharedStorageConstants.INSTANCES);
                    List<LearningNode> leaves = getter.get(SharedStorageConstants.LEAVES);
                    List<LearningNode> layer = getter.get(SharedStorageConstants.LAYER);
                    List<Node> currentTreeNodes;
                    if (layer.size() == 0) {
                        layer =
                                Collections.singletonList(
                                        getter.get(SharedStorageConstants.ROOT_LEARNING_NODE));
                        currentTreeNodes = new ArrayList<>();
                        currentTreeNodes.add(new Node());
                    } else {
                        currentTreeNodes = getter.get(SharedStorageConstants.CURRENT_TREE_NODES);
                    }

                    List<LearningNode> nextLayer =
                            nodeSplitter.split(
                                    currentTreeNodes,
                                    layer,
                                    leaves,
                                    splits.splits,
                                    indices,
                                    instances);
                    setter.set(SharedStorageConstants.LEAVES, leaves);
                    setter.set(SharedStorageConstants.LAYER, nextLayer);
                    setter.set(SharedStorageConstants.CURRENT_TREE_NODES, currentTreeNodes);

                    if (nextLayer.isEmpty()) {
                        // Current tree is finished.
                        setter.set(SharedStorageConstants.NEED_INIT_TREE, true);
                        instanceUpdater.update(
                                getter.get(SharedStorageConstants.PREDS_GRADS_HESSIANS),
                                leaves,
                                indices,
                                instances,
                                d -> setter.set(SharedStorageConstants.PREDS_GRADS_HESSIANS, d),
                                currentTreeNodes);
                        leaves.clear();
                        List<List<Node>> allTrees = getter.get(SharedStorageConstants.ALL_TREES);
                        allTrees.add(currentTreeNodes);

                        setter.set(SharedStorageConstants.LEAVES, new ArrayList<>());
                        setter.set(SharedStorageConstants.SWAPPED_INDICES, new int[0]);
                        setter.set(SharedStorageConstants.ALL_TREES, allTrees);
                    } else {
                        setter.set(SharedStorageConstants.SWAPPED_INDICES, indices);
                        setter.set(SharedStorageConstants.NEED_INIT_TREE, false);
                    }
                });
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Integer> collector)
            throws Exception {
        sharedStorageContext.invoke(
                (getter, setter) -> {
                    setter.set(SharedStorageConstants.PREDS_GRADS_HESSIANS, new PredGradHess[0]);
                    setter.set(SharedStorageConstants.SWAPPED_INDICES, new int[0]);
                    setter.set(SharedStorageConstants.LEAVES, Collections.emptyList());
                    setter.set(SharedStorageConstants.LAYER, Collections.emptyList());
                    setter.set(SharedStorageConstants.CURRENT_TREE_NODES, Collections.emptyList());
                });
    }

    @Override
    public void processElement(StreamRecord<Splits> element) throws Exception {
        splits = element.getValue();
        splitsState.update(Collections.singletonList(splits));
    }

    @Override
    public void close() throws Exception {
        splitsState.clear();
        nodeSplitterState.clear();
        instanceUpdaterState.clear();
        sharedStorageContext.clear();
        super.close();
    }

    @Override
    public void onSharedStorageContextSet(SharedStorageContext context) {
        sharedStorageContext = context;
    }

    @Override
    public String getSharedStorageAccessorID() {
        return sharedStorageAccessorID;
    }
}
