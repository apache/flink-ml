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

import org.apache.flink.ml.common.gbt.datastorage.IterationSharedStorage;

/** Stores keys for shared data stored in {@link IterationSharedStorage}. */
class SharedKeys {
    /**
     * In the iteration, some data needs to be shared between subtasks of different operators within
     * one machine. We use {@link IterationSharedStorage} with co-location mechanism to achieve such
     * purpose. The data is stored in JVM static region, and is accessed through string keys from
     * different operator subtasks. Note the first operator to put the data is the owner of the
     * data, and only the owner can update or delete the data.
     *
     * <p>To be specified, in gradient boosting trees algorithm, there three types of shared data:
     *
     * <ul>
     *   <li>Instances (after binned) and their corresponding predictions, gradients, and hessians
     *       are shared to avoid being stored multiple times or communication.
     *   <li>When initializing every new tree, instances need to be shuffled and split to bagging
     *       instances and non-bagging ones. To reduce the cost, we shuffle instance indices other
     *       than instances. Therefore, the shuffle indices need to be shared to access actual
     *       instances.
     *   <li>After splitting nodes of each layer, instance indices need to be swapped to maintain
     *       {@link LearningNode#slice} and {@link LearningNode#oob}. However, we cannot directly
     *       update the data of shuffle indices above, as it already has an owner. So we use another
     *       key to store instance indices after swapping.
     * </ul>
     */
    static final String INSTANCES = "instances";

    static final String PREDS_GRADS_HESSIANS = "preds_grads_hessians";
    static final String SHUFFLED_INDICES = "shuffled_indices";
    static final String SWAPPED_INDICES = "swapped_indices";

    static final String NODE_FEATURE_PAIRS = "node_feature_pairs";
    static final String LEAVES = "leaves";
    static final String LAYER = "layer";

    static final String ROOT_LEARNING_NODE = "root_learning_node";
    static final String ALL_TREES = "all_trees";
    static final String NEED_INIT_TREE = "need_init_tree";
    static final String HAS_INITED_TREE = "has_inited_tree";

    static final String TRAIN_CONTEXT = "train_context";
}
