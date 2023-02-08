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

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.util.BitSet;

/** Aggregation function for merging histograms. */
public class HistogramAggregateFunction extends RichFlatMapFunction<Histogram, Histogram> {

    private final AggregateFunction<Histogram, Histogram, Histogram> aggregator =
            new Histogram.Aggregator();
    private int numSubtasks;
    private BitSet accepted;
    private Histogram acc = null;

    @Override
    public void flatMap(Histogram value, Collector<Histogram> out) throws Exception {
        if (null == accepted) {
            numSubtasks = getRuntimeContext().getNumberOfParallelSubtasks();
            accepted = new BitSet(numSubtasks);
        }
        int receivedSubtaskId = value.subtaskId;
        Preconditions.checkState(!accepted.get(receivedSubtaskId));
        accepted.set(receivedSubtaskId);
        acc = aggregator.add(value, acc);
        if (numSubtasks == accepted.cardinality()) {
            acc.subtaskId = getRuntimeContext().getIndexOfThisSubtask();
            out.collect(acc);
            accepted = null;
            acc = null;
        }
    }
}
