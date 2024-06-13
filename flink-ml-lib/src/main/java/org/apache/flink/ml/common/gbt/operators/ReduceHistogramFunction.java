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

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

/**
 * This operator reduces histograms for (nodeId, featureId) pairs.
 *
 * <p>The input elements are tuples of (subtask index, (nodeId, featureId) pair index, Histogram).
 * The output elements are tuples of ((nodeId, featureId) pair index, Histogram).
 */
public class ReduceHistogramFunction
        extends RichFlatMapFunction<
                Tuple3<Integer, Integer, Histogram>, Tuple2<Integer, Histogram>> {

    private static final Logger LOG = LoggerFactory.getLogger(ReduceHistogramFunction.class);

    private final Map<Integer, BitSet> pairAccepted = new HashMap<>();
    private final Map<Integer, Histogram> pairHistogram = new HashMap<>();
    private int numSubtasks;

    @Override
    public void open(Configuration parameters) throws Exception {
        numSubtasks = getRuntimeContext().getNumberOfParallelSubtasks();
    }

    @Override
    public void flatMap(
            Tuple3<Integer, Integer, Histogram> value, Collector<Tuple2<Integer, Histogram>> out)
            throws Exception {
        int sourceSubtaskId = value.f0;
        int pairId = value.f1;
        Histogram histogram = value.f2;

        BitSet accepted = pairAccepted.getOrDefault(pairId, new BitSet(numSubtasks));
        if (accepted.isEmpty()) {
            LOG.debug("Received histogram for new pair {}", pairId);
        }
        Preconditions.checkState(!accepted.get(sourceSubtaskId));
        accepted.set(sourceSubtaskId);
        pairAccepted.put(pairId, accepted);

        pairHistogram.compute(pairId, (k, v) -> null == v ? histogram : v.accumulate(histogram));
        if (numSubtasks == accepted.cardinality()) {
            out.collect(Tuple2.of(pairId, pairHistogram.get(pairId)));
            LOG.debug("Output accumulated histogram for pair {}", pairId);
            pairAccepted.remove(pairId);
            pairHistogram.remove(pairId);
        }
    }
}
