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
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.splitter.CategoricalFeatureSplitter;
import org.apache.flink.ml.common.gbt.splitter.ContinuousFeatureSplitter;
import org.apache.flink.ml.common.gbt.splitter.HistogramFeatureSplitter;
import org.apache.flink.util.Preconditions;

class SplitFinder {
    private final HistogramFeatureSplitter[] splitters;
    private final int maxDepth;
    private final int maxNumLeaves;

    public SplitFinder(TrainContext trainContext) {
        FeatureMeta[] featureMetas = trainContext.featureMetas;
        int numFeatures = trainContext.numFeatures;
        splitters = new HistogramFeatureSplitter[numFeatures + 1];
        for (int i = 0; i < numFeatures; ++i) {
            splitters[i] =
                    FeatureMeta.Type.CATEGORICAL == featureMetas[i].type
                            ? new CategoricalFeatureSplitter(
                                    i, featureMetas[i], trainContext.strategy)
                            : new ContinuousFeatureSplitter(
                                    i, featureMetas[i], trainContext.strategy);
        }
        // Adds an addition splitter to obtain the prediction of the node.
        splitters[numFeatures] =
                new ContinuousFeatureSplitter(
                        numFeatures,
                        new FeatureMeta.ContinuousFeatureMeta("SPECIAL", 0, new double[0]),
                        trainContext.strategy);
        maxDepth = trainContext.strategy.maxDepth;
        maxNumLeaves = trainContext.strategy.maxNumLeaves;
    }

    public Split calc(LearningNode node, int featureId, int numLeaves, Histogram histogram) {
        Preconditions.checkState(node.depth < maxDepth || numLeaves + 2 <= maxNumLeaves);
        Preconditions.checkState(histogram.slice.start == 0);
        splitters[featureId].reset(
                histogram.hists, new Slice(0, histogram.hists.length / DataUtils.BIN_SIZE));
        return splitters[featureId].bestSplit();
    }
}
