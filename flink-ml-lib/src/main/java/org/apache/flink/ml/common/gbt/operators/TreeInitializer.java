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

import org.apache.flink.ml.common.gbt.DataUtils;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.TrainContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.IntStream;

class TreeInitializer {
    private static final Logger LOG = LoggerFactory.getLogger(TreeInitializer.class);

    private final int subtaskId;
    private final int numInstances;
    private final int numBaggingInstances;
    private final int[] shuffledIndices;
    private final Random instanceRandomizer;

    public TreeInitializer(TrainContext trainContext) {
        subtaskId = trainContext.subtaskId;
        numInstances = trainContext.numInstances;
        numBaggingInstances = trainContext.numBaggingInstances;
        instanceRandomizer = trainContext.instanceRandomizer;
        shuffledIndices = IntStream.range(0, numInstances).toArray();
    }

    /** Calculate local histograms for nodes in current layer of tree. */
    public void init(int numTrees, Consumer<int[]> shuffledIndicesSetter) {
        LOG.info("subtaskId: {}, {} start", subtaskId, TreeInitializer.class.getSimpleName());
        // Initializes the root node of a new tree when last tree is finalized.
        DataUtils.shuffle(shuffledIndices, instanceRandomizer);
        shuffledIndicesSetter.accept(shuffledIndices);
        LOG.info("subtaskId: {}, initialize {}-th tree", subtaskId, numTrees + 1);
        LOG.info("subtaskId: {}, {} end", this.subtaskId, TreeInitializer.class.getSimpleName());
    }

    public LearningNode getRootLearningNode() {
        return new LearningNode(
                0,
                new Slice(0, numBaggingInstances),
                new Slice(numBaggingInstances, numInstances),
                1);
    }
}
