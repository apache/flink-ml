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

package org.apache.flink.ml.iteration.itcases.operators;

import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;

/** Sources emitting the continuous int sequences */
public class SequenceSource extends RichParallelSourceFunction<Integer> {

    private final int maxValue;

    private final boolean holdAfterMaxValue;

    private final int period;

    private volatile boolean canceled;

    public SequenceSource(int maxValue, boolean holdAfterMaxValue, int period) {
        this.maxValue = maxValue;
        this.holdAfterMaxValue = holdAfterMaxValue;
        this.period = period;
    }

    @Override
    public void run(SourceContext<Integer> ctx) throws Exception {
        for (int i = 0; i < maxValue && !canceled; ++i) {
            ctx.collect(i);
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
}
