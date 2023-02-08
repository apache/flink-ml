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

package org.apache.flink.ml.common.gbt.defs;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.gbt.loss.Loss;

import org.eclipse.collections.api.tuple.primitive.IntIntPair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Stores training state, including static parts and dynamic parts. Static parts won't change across
 * the iteration rounds (except initialization), while dynamic parts are updated on every round.
 *
 * <p>An instance of training states is bound to a subtask id, so the operators accepting training
 * states should be co-located.
 */
public class LocalState implements Serializable {

    public Statics statics;
    public Dynamics dynamics;

    public LocalState(Statics statics, Dynamics dynamics) {
        this.statics = statics;
        this.dynamics = dynamics;
    }

    /** Static part of local state. */
    public static class Statics {

        public int subtaskId;
        public int numSubtasks;
        public GbtParams params;

        public int numInstances;
        public int numBaggingInstances;
        public Random instanceRandomizer;

        public int numFeatures;
        public int numBaggingFeatures;
        public Random featureRandomizer;

        public FeatureMeta[] featureMetas;
        public int[] numFeatureBins;

        public Tuple2<Double, Long> labelSumCount;
        public double prior;
        public Loss loss;
    }

    /** Dynamic part of local state. */
    public static class Dynamics {
        // Root nodes of every tree.
        public List<Node> roots = new ArrayList<>();
        // Initializes a new tree when false, otherwise splits nodes in current layer.
        public boolean inWeakLearner;

        // Nodes to be split in the current layer.
        public List<LearningNode> layer = new ArrayList<>();
        // Node ID and feature ID pairs to be considered in current layer.
        public List<IntIntPair> nodeFeaturePairs = new ArrayList<>();
        // Leaf nodes in the current tree.
        public List<LearningNode> leaves = new ArrayList<>();
    }
}
