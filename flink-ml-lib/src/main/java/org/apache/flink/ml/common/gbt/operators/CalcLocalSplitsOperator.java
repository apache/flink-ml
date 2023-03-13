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
import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.Split;
import org.apache.flink.ml.common.sharedstorage.SharedStorageContext;
import org.apache.flink.ml.common.sharedstorage.SharedStorageStreamOperator;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Calculates best splits from histograms for (nodeId, featureId) pairs.
 *
 * <p>The input elements are tuples of ((nodeId, featureId) pair index, Histogram). The output
 * elements are tuples of (node index, (nodeId, featureId) pair index, Split).
 */
public class CalcLocalSplitsOperator extends AbstractStreamOperator<Tuple3<Integer, Integer, Split>>
        implements OneInputStreamOperator<
                        Tuple2<Integer, Histogram>, Tuple3<Integer, Integer, Split>>,
                SharedStorageStreamOperator {

    private static final Logger LOG = LoggerFactory.getLogger(CalcLocalSplitsOperator.class);
    private static final String SPLIT_FINDER_STATE_NAME = "split_finder";
    private final String sharedStorageAccessorID;
    // States of local data.
    private transient ListStateWithCache<SplitFinder> splitFinderState;
    private transient SplitFinder splitFinder;
    private transient SharedStorageContext sharedStorageContext;

    public CalcLocalSplitsOperator() {
        sharedStorageAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        splitFinderState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(SplitFinder.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        splitFinder =
                OperatorStateUtils.getUniqueElement(splitFinderState, SPLIT_FINDER_STATE_NAME)
                        .orElse(null);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        splitFinderState.snapshotState(context);
    }

    @Override
    public void processElement(StreamRecord<Tuple2<Integer, Histogram>> element) throws Exception {
        if (null == splitFinder) {
            sharedStorageContext.invoke(
                    (getter, setter) -> {
                        splitFinder =
                                new SplitFinder(getter.get(SharedStorageConstants.TRAIN_CONTEXT));
                        splitFinderState.update(Collections.singletonList(splitFinder));
                    });
        }

        Tuple2<Integer, Histogram> value = element.getValue();
        int pairId = value.f0;
        Histogram histogram = value.f1;
        LOG.debug("Received histogram for pairId: {}", pairId);
        sharedStorageContext.invoke(
                (getter, setter) -> {
                    List<LearningNode> layer = getter.get(SharedStorageConstants.LAYER);
                    if (layer.size() == 0) {
                        layer =
                                Collections.singletonList(
                                        getter.get(SharedStorageConstants.ROOT_LEARNING_NODE));
                    }

                    int[] nodeFeaturePairs = getter.get(SharedStorageConstants.NODE_FEATURE_PAIRS);
                    int nodeId = nodeFeaturePairs[2 * pairId];
                    int featureId = nodeFeaturePairs[2 * pairId + 1];
                    LearningNode node = layer.get(nodeId);

                    Split bestSplit =
                            splitFinder.calc(
                                    node,
                                    featureId,
                                    getter.get(SharedStorageConstants.LEAVES).size(),
                                    histogram);
                    output.collect(new StreamRecord<>(Tuple3.of(nodeId, pairId, bestSplit)));
                });
        LOG.debug("Output split for pairId: {}", pairId);
    }

    @Override
    public void close() throws Exception {
        super.close();
        splitFinderState.clear();
    }

    @Override
    public void onSharedStorageContextSet(SharedStorageContext context) {
        this.sharedStorageContext = context;
    }

    @Override
    public String getSharedStorageAccessorID() {
        return sharedStorageAccessorID;
    }
}
