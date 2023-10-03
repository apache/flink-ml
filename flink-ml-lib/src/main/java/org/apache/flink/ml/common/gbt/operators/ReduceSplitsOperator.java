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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.sharedobjects.AbstractSharedObjectsOneInputStreamOperator;
import org.apache.flink.ml.common.sharedobjects.ReadRequest;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.NODE_FEATURE_PAIRS;

/**
 * Reduces best splits for nodes.
 *
 * <p>The input elements are tuples of (node index, (nodeId, featureId) pair index, Split). The
 * output elements are tuples of (node index, Split).
 */
public class ReduceSplitsOperator
        extends AbstractSharedObjectsOneInputStreamOperator<
                Tuple3<Integer, Integer, Split>, Tuple2<Integer, Split>> {

    private static final Logger LOG = LoggerFactory.getLogger(ReduceSplitsOperator.class);

    private Map<Integer, BitSet> nodeFeatureMap;
    private Map<Integer, Split> nodeBestSplit;
    private Map<Integer, Integer> nodeFeatureCounter;

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        nodeFeatureMap = new HashMap<>();
        nodeBestSplit = new HashMap<>();
        nodeFeatureCounter = new HashMap<>();
    }

    @Override
    public void processElement(StreamRecord<Tuple3<Integer, Integer, Split>> element)
            throws Exception {
        if (nodeFeatureMap.isEmpty()) {
            Preconditions.checkState(nodeBestSplit.isEmpty());
            nodeFeatureCounter.clear();
            int[] nodeFeaturePairs = context.read(NODE_FEATURE_PAIRS.nextStep());
            for (int i = 0; i < nodeFeaturePairs.length / 2; i += 1) {
                int nodeId = nodeFeaturePairs[2 * i];
                nodeFeatureCounter.compute(nodeId, (k, v) -> null == v ? 1 : v + 1);
            }
        }

        Tuple3<Integer, Integer, Split> value = element.getValue();
        int nodeId = value.f0;
        int pairId = value.f1;
        Split split = value.f2;
        BitSet featureMap = nodeFeatureMap.getOrDefault(nodeId, new BitSet());
        if (featureMap.isEmpty()) {
            LOG.debug("Received split for new node {}", nodeId);
        }
        int[] nodeFeaturePairs = context.read(NODE_FEATURE_PAIRS.nextStep());
        Preconditions.checkState(nodeId == nodeFeaturePairs[pairId * 2]);
        int featureId = nodeFeaturePairs[pairId * 2 + 1];
        Preconditions.checkState(!featureMap.get(featureId));
        featureMap.set(featureId);
        nodeFeatureMap.put(nodeId, featureMap);

        nodeBestSplit.compute(nodeId, (k, v) -> null == v ? split : v.accumulate(split));
        if (featureMap.cardinality() == nodeFeatureCounter.get(nodeId)) {
            output.collect(new StreamRecord<>(Tuple2.of(nodeId, nodeBestSplit.get(nodeId))));
            LOG.debug("Output accumulated split for node {}", nodeId);
            nodeBestSplit.remove(nodeId);
            nodeFeatureMap.remove(nodeId);
            nodeFeatureCounter.remove(nodeId);
        }
    }

    @Override
    public List<ReadRequest<?>> readRequestsInProcessElement() {
        return Collections.singletonList(NODE_FEATURE_PAIRS.nextStep());
    }
}
