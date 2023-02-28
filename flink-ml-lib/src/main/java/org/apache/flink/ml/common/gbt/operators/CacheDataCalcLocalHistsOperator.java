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

import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.loss.Loss;
import org.apache.flink.ml.common.gbt.typeinfo.BinnedInstanceSerializer;
import org.apache.flink.ml.common.sharedstorage.SharedStorageContext;
import org.apache.flink.ml.common.sharedstorage.SharedStorageStreamOperator;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Calculates local histograms for local data partition. Specifically in the first round, this
 * operator caches all data instances to JVM static region.
 */
public class CacheDataCalcLocalHistsOperator extends AbstractStreamOperator<Histogram>
        implements TwoInputStreamOperator<Row, TrainContext, Histogram>,
                IterationListener<Histogram>,
                SharedStorageStreamOperator {

    private static final String TREE_INITIALIZER_STATE_NAME = "tree_initializer";
    private static final String HIST_BUILDER_STATE_NAME = "hist_builder";

    private final GbtParams gbtParams;
    private final String sharedStorageAccessorID;

    // States of local data.
    private transient ListStateWithCache<BinnedInstance> instancesCollecting;
    private transient ListStateWithCache<TreeInitializer> treeInitializerState;
    private transient TreeInitializer treeInitializer;
    private transient ListStateWithCache<HistBuilder> histBuilderState;
    private transient HistBuilder histBuilder;
    private transient SharedStorageContext sharedStorageContext;

    public CacheDataCalcLocalHistsOperator(GbtParams gbtParams) {
        super();
        this.gbtParams = gbtParams;
        sharedStorageAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);

        instancesCollecting =
                new ListStateWithCache<>(
                        BinnedInstanceSerializer.INSTANCE,
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        treeInitializerState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(TreeInitializer.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        treeInitializer =
                OperatorStateUtils.getUniqueElement(
                                treeInitializerState, TREE_INITIALIZER_STATE_NAME)
                        .orElse(null);
        histBuilderState =
                new ListStateWithCache<>(
                        new KryoSerializer<>(HistBuilder.class, getExecutionConfig()),
                        getContainingTask(),
                        getRuntimeContext(),
                        context,
                        getOperatorID());
        histBuilder =
                OperatorStateUtils.getUniqueElement(histBuilderState, HIST_BUILDER_STATE_NAME)
                        .orElse(null);

        sharedStorageContext.initializeState(this, getRuntimeContext(), context);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        instancesCollecting.snapshotState(context);
        treeInitializerState.snapshotState(context);
        histBuilderState.snapshotState(context);
        sharedStorageContext.snapshotState(context);
    }

    @Override
    public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
        Row row = streamRecord.getValue();
        BinnedInstance instance = new BinnedInstance();
        instance.weight = 1.;
        instance.label = row.getFieldAs(gbtParams.labelCol);

        if (gbtParams.isInputVector) {
            Vector vec = row.getFieldAs(gbtParams.featuresCols[0]);
            SparseVector sv = vec.toSparse();
            instance.featureIds = sv.indices.length == sv.size() ? null : sv.indices;
            instance.featureValues = Arrays.stream(sv.values).mapToInt(d -> (int) d).toArray();
        } else {
            instance.featureValues =
                    Arrays.stream(gbtParams.featuresCols)
                            .mapToInt(col -> ((Number) row.getFieldAs(col)).intValue())
                            .toArray();
        }
        instancesCollecting.add(instance);
    }

    @Override
    public void processElement2(StreamRecord<TrainContext> streamRecord) throws Exception {
        TrainContext rawTrainContext = streamRecord.getValue();
        sharedStorageContext.invoke(
                (getter, setter) ->
                        setter.set(SharedStorageConstants.TRAIN_CONTEXT, rawTrainContext));
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Histogram> out) throws Exception {
        if (0 == epochWatermark) {
            // Initializes local state in first round.
            sharedStorageContext.invoke(
                    (getter, setter) -> {
                        BinnedInstance[] instances =
                                (BinnedInstance[])
                                        IteratorUtils.toArray(
                                                instancesCollecting.get().iterator(),
                                                BinnedInstance.class);
                        setter.set(SharedStorageConstants.INSTANCES, instances);
                        instancesCollecting.clear();

                        TrainContext rawTrainContext =
                                getter.get(SharedStorageConstants.TRAIN_CONTEXT);
                        TrainContext trainContext =
                                new TrainContextInitializer(gbtParams)
                                        .init(
                                                rawTrainContext,
                                                getRuntimeContext().getIndexOfThisSubtask(),
                                                getRuntimeContext().getNumberOfParallelSubtasks(),
                                                instances);
                        setter.set(SharedStorageConstants.TRAIN_CONTEXT, trainContext);

                        treeInitializer = new TreeInitializer(trainContext);
                        treeInitializerState.update(Collections.singletonList(treeInitializer));
                        histBuilder = new HistBuilder(trainContext);
                        histBuilderState.update(Collections.singletonList(histBuilder));
                    });
        }

        sharedStorageContext.invoke(
                (getter, setter) -> {
                    TrainContext trainContext = getter.get(SharedStorageConstants.TRAIN_CONTEXT);
                    Preconditions.checkArgument(
                            getRuntimeContext().getIndexOfThisSubtask() == trainContext.subtaskId);
                    BinnedInstance[] instances = getter.get(SharedStorageConstants.INSTANCES);
                    PredGradHess[] pgh = getter.get(SharedStorageConstants.PREDS_GRADS_HESSIANS);
                    // In the first round, use prior as the predictions.
                    if (0 == pgh.length) {
                        pgh = new PredGradHess[instances.length];
                        double prior = trainContext.prior;
                        Loss loss = trainContext.loss;
                        for (int i = 0; i < instances.length; i += 1) {
                            double label = instances[i].label;
                            pgh[i] =
                                    new PredGradHess(
                                            prior,
                                            loss.gradient(prior, label),
                                            loss.hessian(prior, label));
                        }
                    }

                    boolean needInitTree = getter.get(SharedStorageConstants.NEED_INIT_TREE);
                    int[] indices;
                    List<LearningNode> layer;
                    if (needInitTree) {
                        // When last tree is finished, initializes a new tree, and shuffle instance
                        // indices.
                        treeInitializer.init(
                                d -> setter.set(SharedStorageConstants.SHUFFLED_INDICES, d));
                        LearningNode rootLearningNode = treeInitializer.getRootLearningNode();
                        indices = getter.get(SharedStorageConstants.SHUFFLED_INDICES);
                        layer = Collections.singletonList(rootLearningNode);
                        setter.set(SharedStorageConstants.ROOT_LEARNING_NODE, rootLearningNode);
                        setter.set(SharedStorageConstants.HAS_INITED_TREE, true);
                    } else {
                        // Otherwise, uses the swapped instance indices.
                        indices = getter.get(SharedStorageConstants.SWAPPED_INDICES);
                        layer = getter.get(SharedStorageConstants.LAYER);
                        setter.set(SharedStorageConstants.SHUFFLED_INDICES, new int[0]);
                        setter.set(SharedStorageConstants.HAS_INITED_TREE, false);
                    }

                    Histogram localHists =
                            histBuilder.build(
                                    layer,
                                    indices,
                                    instances,
                                    pgh,
                                    d -> setter.set(SharedStorageConstants.NODE_FEATURE_PAIRS, d));
                    out.collect(localHists);
                });
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Histogram> collector)
            throws Exception {
        instancesCollecting.clear();
        treeInitializerState.clear();
        histBuilderState.clear();

        sharedStorageContext.invoke(
                (getter, setter) -> {
                    setter.set(SharedStorageConstants.INSTANCES, new BinnedInstance[0]);
                    setter.set(SharedStorageConstants.SHUFFLED_INDICES, new int[0]);
                    setter.set(SharedStorageConstants.NODE_FEATURE_PAIRS, new int[0]);
                });
    }

    @Override
    public void close() throws Exception {
        instancesCollecting.clear();
        treeInitializerState.clear();
        histBuilderState.clear();
        sharedStorageContext.clear();
        super.close();
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
