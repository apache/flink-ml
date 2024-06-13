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

import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.BoostingStrategy;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.typeinfo.BinnedInstanceSerializer;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.common.sharedobjects.AbstractSharedObjectsTwoInputStreamOperator;
import org.apache.flink.ml.common.sharedobjects.ReadRequest;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.ALL_TREES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.HAS_INITED_TREE;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.INSTANCES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.LAYER;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.NEED_INIT_TREE;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.NODE_FEATURE_PAIRS;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.PREDS_GRADS_HESSIANS;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.ROOT_LEARNING_NODE;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.SHUFFLED_INDICES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.SWAPPED_INDICES;
import static org.apache.flink.ml.common.gbt.operators.SharedObjectsConstants.TRAIN_CONTEXT;

/**
 * Calculates local histograms for local data partition.
 *
 * <p>This operator only has input elements in the first round, including data instances and raw
 * training context. There will be no input elements in other rounds. The output elements are tuples
 * of (subtask index, (nodeId, featureId) pair index, Histogram).
 */
public class CacheDataCalcLocalHistsOperator
        extends AbstractSharedObjectsTwoInputStreamOperator<
                Row, TrainContext, Tuple3<Integer, Integer, Histogram>>
        implements IterationListener<Tuple3<Integer, Integer, Histogram>> {

    private static final String TREE_INITIALIZER_STATE_NAME = "tree_initializer";
    private static final String HIST_BUILDER_STATE_NAME = "hist_builder";

    private final BoostingStrategy strategy;

    // States of local data.
    private transient TrainContext rawTrainContext;
    private transient ListStateWithCache<BinnedInstance> instancesCollecting;
    private transient ListStateWithCache<TreeInitializer> treeInitializerState;
    private transient TreeInitializer treeInitializer;
    private transient ListStateWithCache<HistBuilder> histBuilderState;
    private transient HistBuilder histBuilder;

    public CacheDataCalcLocalHistsOperator(BoostingStrategy strategy) {
        super();
        this.strategy = strategy;
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
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        instancesCollecting.snapshotState(context);
        treeInitializerState.snapshotState(context);
        histBuilderState.snapshotState(context);
    }

    @Override
    public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
        Row row = streamRecord.getValue();
        BinnedInstance instance = new BinnedInstance();
        instance.weight = 1.;
        instance.label = row.<Number>getFieldAs(strategy.labelCol).doubleValue();

        if (strategy.isInputVector) {
            Vector vec = row.getFieldAs(strategy.featuresCols[0]);
            SparseVector sv = vec.toSparse();
            instance.featureIds = sv.indices.length == sv.size() ? null : sv.indices;
            instance.featureValues = Arrays.stream(sv.values).mapToInt(d -> (int) d).toArray();
        } else {
            instance.featureValues =
                    Arrays.stream(strategy.featuresCols)
                            .mapToInt(col -> ((Number) row.getFieldAs(col)).intValue())
                            .toArray();
        }
        instancesCollecting.add(instance);
    }

    @Override
    public List<ReadRequest<?>> readRequestsInProcessElement1() {
        return Collections.emptyList();
    }

    @Override
    public void processElement2(StreamRecord<TrainContext> streamRecord) {
        rawTrainContext = streamRecord.getValue();
    }

    @Override
    public List<ReadRequest<?>> readRequestsInProcessElement2() {
        return Collections.emptyList();
    }

    public void onEpochWatermarkIncremented(
            int epochWatermark, Context c, Collector<Tuple3<Integer, Integer, Histogram>> out)
            throws Exception {
        if (0 == epochWatermark) {
            // Initializes local state in first round.
            BinnedInstance[] instances =
                    (BinnedInstance[])
                            IteratorUtils.toArray(
                                    instancesCollecting.get().iterator(), BinnedInstance.class);
            context.write(INSTANCES, instances);
            instancesCollecting.clear();

            TrainContext trainContext =
                    new TrainContextInitializer(strategy)
                            .init(
                                    rawTrainContext,
                                    getRuntimeContext().getIndexOfThisSubtask(),
                                    getRuntimeContext().getNumberOfParallelSubtasks(),
                                    instances);
            context.write(TRAIN_CONTEXT, trainContext);

            treeInitializer = new TreeInitializer(trainContext);
            treeInitializerState.update(Collections.singletonList(treeInitializer));
            histBuilder = new HistBuilder(trainContext);
            histBuilderState.update(Collections.singletonList(histBuilder));

        } else {
            context.renew(TRAIN_CONTEXT);
            context.renew(INSTANCES);
        }

        TrainContext trainContext = context.read(TRAIN_CONTEXT.sameStep());
        Preconditions.checkArgument(
                getRuntimeContext().getIndexOfThisSubtask() == trainContext.subtaskId);
        BinnedInstance[] instances = context.read(INSTANCES.sameStep());

        double[] pgh = new double[0];
        boolean needInitTree = true;
        int numTrees = 0;
        if (epochWatermark > 0) {
            pgh = context.read(PREDS_GRADS_HESSIANS.prevStep());
            needInitTree = context.read(NEED_INIT_TREE.prevStep());
            numTrees = context.read(ALL_TREES.prevStep()).size();
        }
        // In the first round, use prior as the predictions.
        if (0 == pgh.length) {
            pgh = new double[instances.length * 3];
            double prior = trainContext.prior;
            LossFunc loss = trainContext.loss;
            for (int i = 0; i < instances.length; i += 1) {
                double label = instances[i].label;
                pgh[3 * i] = prior;
                pgh[3 * i + 1] = loss.gradient(prior, label);
                pgh[3 * i + 2] = loss.hessian(prior, label);
            }
        }

        int[] indices;
        List<LearningNode> layer;
        if (needInitTree) {
            // When last tree is finished, initializes a new tree, and shuffle instance
            // indices.
            treeInitializer.init(numTrees, d -> context.write(SHUFFLED_INDICES, d));
            LearningNode rootLearningNode = treeInitializer.getRootLearningNode();
            indices = context.read(SHUFFLED_INDICES.sameStep());
            layer = Collections.singletonList(rootLearningNode);
            context.write(ROOT_LEARNING_NODE, rootLearningNode);
            context.write(HAS_INITED_TREE, true);
        } else {
            // Otherwise, uses the swapped instance indices.
            indices = context.read(SWAPPED_INDICES.prevStep());
            layer = context.read(LAYER.prevStep());
            context.write(SHUFFLED_INDICES, new int[0]);
            context.write(HAS_INITED_TREE, false);
            context.renew(ROOT_LEARNING_NODE);
        }

        histBuilder.build(
                layer, indices, instances, pgh, d -> context.write(NODE_FEATURE_PAIRS, d), out);
    }

    @Override
    public void onIterationTerminated(
            Context c, Collector<Tuple3<Integer, Integer, Histogram>> collector) {
        instancesCollecting.clear();
        treeInitializerState.clear();
        histBuilderState.clear();

        context.write(INSTANCES, new BinnedInstance[0]);
        context.write(SHUFFLED_INDICES, new int[0]);
        context.write(NODE_FEATURE_PAIRS, new int[0]);
    }

    @Override
    public void close() throws Exception {
        instancesCollecting.clear();
        treeInitializerState.clear();
        histBuilderState.clear();
        super.close();
    }
}
