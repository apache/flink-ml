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

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeutils.base.BooleanSerializer;
import org.apache.flink.api.common.typeutils.base.GenericArraySerializer;
import org.apache.flink.api.common.typeutils.base.ListSerializer;
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.datastorage.IterationSharedStorage;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.typeinfo.LearningNodeSerializer;
import org.apache.flink.ml.common.gbt.typeinfo.NodeSerializer;
import org.apache.flink.ml.common.gbt.typeinfo.PredGradHessSerializer;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Post-process after global splits obtained, including split instances to left or child nodes, and
 * update instances scores after a tree is complete.
 */
public class PostSplitsOperator extends AbstractStreamOperator<Integer>
        implements OneInputStreamOperator<Splits, Integer>, IterationListener<Integer> {

    private static final String SPLITS_STATE_NAME = "splits";
    private static final String NODE_SPLITTER_STATE_NAME = "node_splitter";
    private static final String INSTANCE_UPDATER_STATE_NAME = "instance_updater";
    private static final String CURRENT_TREE_NODES_STATE_NAME = "current_tree_nodes";

    private final IterationID iterationID;

    private IterationSharedStorage.Reader<BinnedInstance[]> instancesReader;
    private IterationSharedStorage.Writer<PredGradHess[]> pghWriter;
    private IterationSharedStorage.Reader<int[]> shuffledIndicesReader;
    private IterationSharedStorage.Writer<int[]> swappedIndicesWriter;

    private transient ListState<Splits> splits;
    private transient ListState<NodeSplitter> nodeSplitter;
    private transient ListState<InstanceUpdater> instanceUpdater;
    private IterationSharedStorage.Writer<List<LearningNode>> leavesWriter;
    private IterationSharedStorage.Writer<List<LearningNode>> layerWriter;
    private IterationSharedStorage.Reader<LearningNode> rootLearningNodeReader;
    private IterationSharedStorage.Writer<List<List<Node>>> allTreesWriter;
    private IterationSharedStorage.Writer<List<Node>> currentTreeNodesWriter;
    private IterationSharedStorage.Writer<Boolean> needInitTreeWriter;
    private IterationSharedStorage.Reader<TrainContext> trainContextReader;

    public PostSplitsOperator(IterationID iterationID) {
        this.iterationID = iterationID;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);

        splits =
                context.getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>(SPLITS_STATE_NAME, Splits.class));
        nodeSplitter =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        NODE_SPLITTER_STATE_NAME, NodeSplitter.class));
        instanceUpdater =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        INSTANCE_UPDATER_STATE_NAME, InstanceUpdater.class));

        int subtaskId = getRuntimeContext().getIndexOfThisSubtask();
        pghWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.PREDS_GRADS_HESSIANS,
                        getOperatorID(),
                        new GenericArraySerializer<>(
                                PredGradHess.class, PredGradHessSerializer.INSTANCE),
                        new PredGradHess[0]);
        pghWriter.initializeState(context);
        swappedIndicesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.SWAPPED_INDICES,
                        getOperatorID(),
                        IntPrimitiveArraySerializer.INSTANCE,
                        new int[0]);
        swappedIndicesWriter.initializeState(context);
        leavesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.LEAVES,
                        getOperatorID(),
                        new ListSerializer<>(LearningNodeSerializer.INSTANCE),
                        new ArrayList<>());
        leavesWriter.initializeState(context);

        layerWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.LAYER,
                        getOperatorID(),
                        new ListSerializer<>(LearningNodeSerializer.INSTANCE),
                        new ArrayList<>());
        layerWriter.initializeState(context);

        allTreesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.ALL_TREES,
                        getOperatorID(),
                        new ListSerializer<>(new ListSerializer<>(NodeSerializer.INSTANCE)),
                        new ArrayList<>());
        allTreesWriter.initializeState(context);

        needInitTreeWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.NEED_INIT_TREE,
                        getOperatorID(),
                        BooleanSerializer.INSTANCE,
                        true);
        needInitTreeWriter.initializeState(context);

        currentTreeNodesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        CURRENT_TREE_NODES_STATE_NAME,
                        getOperatorID(),
                        new ListSerializer<>(NodeSerializer.INSTANCE),
                        new ArrayList<>());
        currentTreeNodesWriter.initializeState(context);

        instancesReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, SharedKeys.INSTANCES);
        shuffledIndicesReader =
                IterationSharedStorage.getReader(
                        iterationID, subtaskId, SharedKeys.SHUFFLED_INDICES);
        rootLearningNodeReader =
                IterationSharedStorage.getReader(
                        iterationID, subtaskId, SharedKeys.ROOT_LEARNING_NODE);
        trainContextReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, SharedKeys.TRAIN_CONTEXT);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        pghWriter.snapshotState(context);
        swappedIndicesWriter.snapshotState(context);
        leavesWriter.snapshotState(context);
        needInitTreeWriter.snapshotState(context);
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Integer> collector) throws Exception {
        if (0 == epochWatermark) {
            nodeSplitter.update(
                    Collections.singletonList(new NodeSplitter(trainContextReader.get())));
            instanceUpdater.update(
                    Collections.singletonList(new InstanceUpdater(trainContextReader.get())));
        }

        int[] indices = swappedIndicesWriter.get();
        if (0 == indices.length) {
            indices = shuffledIndicesReader.get().clone();
        }

        BinnedInstance[] instances = instancesReader.get();
        List<LearningNode> leaves = leavesWriter.get();
        List<LearningNode> layer = layerWriter.get();
        List<Node> currentTreeNodes;
        if (layer.size() == 0) {
            layer = Collections.singletonList(rootLearningNodeReader.get());
            currentTreeNodes = new ArrayList<>();
            currentTreeNodes.add(new Node());
        } else {
            currentTreeNodes = currentTreeNodesWriter.get();
        }

        List<LearningNode> nextLayer =
                OperatorStateUtils.getUniqueElement(nodeSplitter, NODE_SPLITTER_STATE_NAME)
                        .get()
                        .split(
                                currentTreeNodes,
                                layer,
                                leaves,
                                OperatorStateUtils.getUniqueElement(splits, SPLITS_STATE_NAME)
                                        .get()
                                        .splits,
                                indices,
                                instances);
        leavesWriter.set(leaves);
        layerWriter.set(nextLayer);
        currentTreeNodesWriter.set(currentTreeNodes);

        if (nextLayer.isEmpty()) {
            needInitTreeWriter.set(true);
            OperatorStateUtils.getUniqueElement(instanceUpdater, INSTANCE_UPDATER_STATE_NAME)
                    .get()
                    .update(leaves, indices, instances, pghWriter::set, currentTreeNodes);
            leaves.clear();
            List<List<Node>> allTrees = allTreesWriter.get();
            allTrees.add(currentTreeNodes);

            leavesWriter.set(new ArrayList<>());
            swappedIndicesWriter.set(new int[0]);
            allTreesWriter.set(allTrees);
        } else {
            swappedIndicesWriter.set(indices);
            needInitTreeWriter.set(false);
        }
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Integer> collector) {
        pghWriter.set(new PredGradHess[0]);
        swappedIndicesWriter.set(new int[0]);
        leavesWriter.set(Collections.emptyList());
        layerWriter.set(Collections.emptyList());
        currentTreeNodesWriter.set(Collections.emptyList());
    }

    @Override
    public void processElement(StreamRecord<Splits> element) throws Exception {
        splits.update(Collections.singletonList(element.getValue()));
    }

    @Override
    public void close() throws Exception {
        pghWriter.remove();
        swappedIndicesWriter.remove();
        leavesWriter.remove();
        layerWriter.remove();
        allTreesWriter.remove();
        currentTreeNodesWriter.remove();
        needInitTreeWriter.remove();
        super.close();
    }
}
