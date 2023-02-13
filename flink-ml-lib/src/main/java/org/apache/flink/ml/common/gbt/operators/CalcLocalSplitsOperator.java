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
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.datastorage.IterationSharedStorage;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

import java.util.Collections;
import java.util.List;

/** Calculates local splits for assigned (nodeId, featureId) pairs. */
public class CalcLocalSplitsOperator extends AbstractStreamOperator<Splits>
        implements OneInputStreamOperator<Histogram, Splits>, IterationListener<Splits> {

    private static final String CALC_BEST_SPLIT_STATE_NAME = "split_finder";
    private static final String HISTOGRAM_STATE_NAME = "histogram";

    private final IterationID iterationID;

    private transient ListState<SplitFinder> splitFinder;
    private transient ListState<Histogram> histogram;
    private IterationSharedStorage.Reader<int[]> nodeFeaturePairsReader;
    private IterationSharedStorage.Reader<List<LearningNode>> leavesReader;
    private IterationSharedStorage.Reader<List<LearningNode>> layerReader;
    private IterationSharedStorage.Reader<LearningNode> rootLearningNodeReader;
    private IterationSharedStorage.Reader<TrainContext> trainContextReader;

    public CalcLocalSplitsOperator(IterationID iterationID) {
        this.iterationID = iterationID;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        splitFinder =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        CALC_BEST_SPLIT_STATE_NAME, SplitFinder.class));
        histogram =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(HISTOGRAM_STATE_NAME, Histogram.class));

        int subtaskId = getRuntimeContext().getIndexOfThisSubtask();
        nodeFeaturePairsReader =
                IterationSharedStorage.getReader(
                        iterationID, subtaskId, SharedKeys.NODE_FEATURE_PAIRS);
        leavesReader = IterationSharedStorage.getReader(iterationID, subtaskId, SharedKeys.LEAVES);
        layerReader = IterationSharedStorage.getReader(iterationID, subtaskId, SharedKeys.LAYER);
        rootLearningNodeReader =
                IterationSharedStorage.getReader(
                        iterationID, subtaskId, SharedKeys.ROOT_LEARNING_NODE);
        trainContextReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, SharedKeys.TRAIN_CONTEXT);
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Splits> collector) throws Exception {
        if (0 == epochWatermark) {
            splitFinder.update(
                    Collections.singletonList(new SplitFinder(trainContextReader.get())));
        }

        List<LearningNode> layer = layerReader.get();
        if (layer.size() == 0) {
            layer = Collections.singletonList(rootLearningNodeReader.get());
        }

        Splits splits =
                OperatorStateUtils.getUniqueElement(splitFinder, CALC_BEST_SPLIT_STATE_NAME)
                        .get()
                        .calc(
                                layer,
                                nodeFeaturePairsReader.get(),
                                leavesReader.get().size(),
                                OperatorStateUtils.getUniqueElement(histogram, HISTOGRAM_STATE_NAME)
                                        .get());
        collector.collect(splits);
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Splits> collector) {}

    @Override
    public void processElement(StreamRecord<Histogram> element) throws Exception {
        histogram.update(Collections.singletonList(element.getValue()));
    }
}
