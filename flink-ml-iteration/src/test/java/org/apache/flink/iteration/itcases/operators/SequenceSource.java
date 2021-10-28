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

package org.apache.flink.iteration.itcases.operators;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;

import java.util.Collections;

/** Sources emitting the continuous int sequences. */
public class SequenceSource extends RichParallelSourceFunction<Integer>
        implements CheckpointedFunction {

    private final int maxValue;

    private final boolean holdAfterMaxValue;

    private final int period;

    private volatile boolean canceled;

    private int next;

    private ListState<Integer> nextState;

    public SequenceSource(int maxValue, boolean holdAfterMaxValue, int period) {
        this.maxValue = maxValue;
        this.holdAfterMaxValue = holdAfterMaxValue;
        this.period = period;
    }

    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext)
            throws Exception {
        nextState =
                functionInitializationContext
                        .getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("next", Integer.class));
        next = OperatorStateUtils.getUniqueElement(nextState, "next").orElse(0);
    }

    @Override
    public void run(SourceContext<Integer> ctx) throws Exception {
        while (next < maxValue && !canceled) {
            synchronized (ctx.getCheckpointLock()) {
                ctx.collect(next++);
            }

            if (period > 0) {
                Thread.sleep(period);
            }
        }

        if (holdAfterMaxValue) {
            while (!canceled) {
                Thread.sleep(5000);
            }
        }
    }

    @Override
    public void cancel() {
        canceled = true;
    }

    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        nextState.clear();
        nextState.update(Collections.singletonList(next));
    }
}
