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

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeutils.base.BooleanSerializer;
import org.apache.flink.api.common.typeutils.base.GenericArraySerializer;
import org.apache.flink.api.common.typeutils.base.ListSerializer;
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.ml.common.gbt.GBTRunner;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Node;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.typeinfo.BinnedInstanceSerializer;
import org.apache.flink.ml.common.gbt.typeinfo.LearningNodeSerializer;
import org.apache.flink.ml.common.gbt.typeinfo.NodeSerializer;
import org.apache.flink.ml.common.sharedobjects.ItemDescriptor;
import org.apache.flink.ml.common.sharedobjects.SharedObjectsUtils;
import org.apache.flink.ml.linalg.typeinfo.OptimizedDoublePrimitiveArraySerializer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Stores constants used for {@link SharedObjectsUtils} in {@link GBTRunner}.
 *
 * <p>In the iteration, some data needs to be shared and accessed between subtasks of different
 * operators within one JVM to reduce memory footprint and communication cost. We use {@link
 * SharedObjectsUtils} with co-location mechanism to achieve such purpose.
 *
 * <p>All shared data items have corresponding {@link ItemDescriptor}s, and can be read/written
 * through {@link ItemDescriptor}s from different operator subtasks. Note that every shared item has
 * an owner, and the owner can set new values and snapshot the item.
 *
 * <p>This class records all {@link ItemDescriptor}s used in {@link GBTRunner} and their owners.
 */
@Internal
public class SharedObjectsConstants {

    /** Instances (after binned). */
    static final ItemDescriptor<BinnedInstance[]> INSTANCES =
            ItemDescriptor.of(
                    "instances",
                    new GenericArraySerializer<>(
                            BinnedInstance.class, BinnedInstanceSerializer.INSTANCE),
                    new BinnedInstance[0]);

    /**
     * (prediction, gradient, and hessian) of instances, sharing same indexing with {@link
     * #INSTANCES}.
     */
    static final ItemDescriptor<double[]> PREDS_GRADS_HESSIANS =
            ItemDescriptor.of(
                    "preds_grads_hessians",
                    new OptimizedDoublePrimitiveArraySerializer(),
                    new double[0]);

    /** Shuffle indices of instances used after every new tree just initialized. */
    static final ItemDescriptor<int[]> SHUFFLED_INDICES =
            ItemDescriptor.of("shuffled_indices", IntPrimitiveArraySerializer.INSTANCE, new int[0]);

    /** Swapped indices of instances used when {@link #SHUFFLED_INDICES} not applicable. */
    static final ItemDescriptor<int[]> SWAPPED_INDICES =
            ItemDescriptor.of("swapped_indices", IntPrimitiveArraySerializer.INSTANCE, new int[0]);

    /** (nodeId, featureId) pairs used to calculate histograms. */
    static final ItemDescriptor<int[]> NODE_FEATURE_PAIRS =
            ItemDescriptor.of(
                    "node_feature_pairs", IntPrimitiveArraySerializer.INSTANCE, new int[0]);

    /** Leaves nodes of current working tree. */
    static final ItemDescriptor<List<LearningNode>> LEAVES =
            ItemDescriptor.of(
                    "leaves",
                    new ListSerializer<>(LearningNodeSerializer.INSTANCE),
                    new ArrayList<>());

    /** Nodes in current layer of current working tree. */
    static final ItemDescriptor<List<LearningNode>> LAYER =
            ItemDescriptor.of(
                    "layer",
                    new ListSerializer<>(LearningNodeSerializer.INSTANCE),
                    new ArrayList<>());

    /** The root node when initializing a new tree. */
    static final ItemDescriptor<LearningNode> ROOT_LEARNING_NODE =
            ItemDescriptor.of(
                    "root_learning_node", LearningNodeSerializer.INSTANCE, new LearningNode());

    /** All finished trees. */
    static final ItemDescriptor<List<List<Node>>> ALL_TREES =
            ItemDescriptor.of(
                    "all_trees",
                    new ListSerializer<>(new ListSerializer<>(NodeSerializer.INSTANCE)),
                    new ArrayList<>());

    /** Nodes in current working tree. */
    static final ItemDescriptor<List<Node>> CURRENT_TREE_NODES =
            ItemDescriptor.of(
                    "current_tree_nodes",
                    new ListSerializer<>(NodeSerializer.INSTANCE),
                    new ArrayList<>());

    /** Indicates the necessity of initializing a new tree. */
    static final ItemDescriptor<Boolean> NEED_INIT_TREE =
            ItemDescriptor.of("need_init_tree", BooleanSerializer.INSTANCE, true);

    /** Data items owned by the `PostSplits` operator. */
    public static final List<ItemDescriptor<?>> OWNED_BY_POST_SPLITS_OP =
            Arrays.asList(
                    PREDS_GRADS_HESSIANS,
                    SWAPPED_INDICES,
                    LEAVES,
                    LAYER,
                    ALL_TREES,
                    CURRENT_TREE_NODES,
                    NEED_INIT_TREE);

    /** Indicate a new tree has been initialized. */
    static final ItemDescriptor<Boolean> HAS_INITED_TREE =
            ItemDescriptor.of("has_inited_tree", BooleanSerializer.INSTANCE, false);

    /** Training context. */
    static final ItemDescriptor<TrainContext> TRAIN_CONTEXT =
            ItemDescriptor.of(
                    "train_context",
                    new KryoSerializer<>(TrainContext.class, new ExecutionConfig()),
                    new TrainContext());

    /** Data items owned by the `CacheDataCalcLocalHists` operator. */
    public static final List<ItemDescriptor<?>> OWNED_BY_CACHE_DATA_CALC_LOCAL_HISTS_OP =
            Arrays.asList(
                    INSTANCES,
                    SHUFFLED_INDICES,
                    NODE_FEATURE_PAIRS,
                    ROOT_LEARNING_NODE,
                    HAS_INITED_TREE,
                    TRAIN_CONTEXT);
}
