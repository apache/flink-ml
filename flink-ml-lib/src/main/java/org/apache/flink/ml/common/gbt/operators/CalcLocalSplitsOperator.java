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
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import java.util.Collections;

/** Calculates local splits for assigned (nodeId, featureId) pairs. */
public class CalcLocalSplitsOperator extends AbstractStreamOperator<Splits>
        implements TwoInputStreamOperator<LocalState, Histogram, Splits>,
                IterationListener<Splits> {

    private static final String LOCAL_STATE_STATE_NAME = "local_state";
    private static final String CALC_BEST_SPLIT_STATE_NAME = "split_finder";
    private static final String HISTOGRAM_STATE_NAME = "histogram";

    private final OutputTag<LocalState> stateOutputTag;

    private transient ListState<LocalState> localState;
    private transient ListState<SplitFinder> splitFinder;
    private transient ListState<Histogram> histogram;

    public CalcLocalSplitsOperator(OutputTag<LocalState> stateOutputTag) {
        this.stateOutputTag = stateOutputTag;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        localState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        LOCAL_STATE_STATE_NAME, LocalState.class));
        splitFinder =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        CALC_BEST_SPLIT_STATE_NAME, SplitFinder.class));
        histogram =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(HISTOGRAM_STATE_NAME, Histogram.class));
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Splits> collector) throws Exception {
        LocalState localStateValue =
                OperatorStateUtils.getUniqueElement(localState, LOCAL_STATE_STATE_NAME).get();
        if (0 == epochWatermark) {
            splitFinder.update(Collections.singletonList(new SplitFinder(localStateValue.statics)));
        }
        Splits splits =
                OperatorStateUtils.getUniqueElement(splitFinder, CALC_BEST_SPLIT_STATE_NAME)
                        .get()
                        .calc(
                                localStateValue.dynamics.layer,
                                localStateValue.dynamics.nodeFeaturePairs,
                                localStateValue.dynamics.leaves,
                                OperatorStateUtils.getUniqueElement(histogram, HISTOGRAM_STATE_NAME)
                                        .get());
        collector.collect(splits);
        context.output(stateOutputTag, localStateValue);
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Splits> collector) {}

    @Override
    public void processElement1(StreamRecord<LocalState> element) throws Exception {
        localState.update(Collections.singletonList(element.getValue()));
    }

    @Override
    public void processElement2(StreamRecord<Histogram> element) throws Exception {
        histogram.update(Collections.singletonList(element.getValue()));
    }
}
