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

import org.apache.flink.iteration.IterationListener;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.common.sharedobjects.AbstractSharedObjectsOneInputStreamOperator;
import org.apache.flink.ml.common.sharedobjects.ReadRequest;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import java.util.Collections;
import java.util.List;

import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.ALL_TREES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.TRAIN_CONTEXT;

/** Determines whether to terminated training. */
public class TerminationOperator
        extends AbstractSharedObjectsOneInputStreamOperator<Integer, Integer>
        implements IterationListener<GBTModelData> {

    private final OutputTag<GBTModelData> modelDataOutputTag;

    public TerminationOperator(OutputTag<GBTModelData> modelDataOutputTag) {
        this.modelDataOutputTag = modelDataOutputTag;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
    }

    @Override
    public void processElement(StreamRecord<Integer> element) throws Exception {}

    @Override
    public List<ReadRequest<?>> readRequestsInProcessElement() {
        return Collections.emptyList();
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context c, Collector<GBTModelData> collector) {
        boolean terminated =
                context.read(ALL_TREES.sameStep()).size()
                        == context.read(TRAIN_CONTEXT.sameStep()).strategy.maxIter;
        // TODO: Add validation error rate
        if (!terminated) {
            output.collect(new StreamRecord<>(0));
        }
    }

    @Override
    public void onIterationTerminated(Context c, Collector<GBTModelData> collector) {
        if (0 == getRuntimeContext().getIndexOfThisSubtask()) {
            c.output(
                    modelDataOutputTag,
                    GBTModelData.from(
                            context.read(TRAIN_CONTEXT.prevStep()),
                            context.read(ALL_TREES.prevStep())));
        }
    }
}
