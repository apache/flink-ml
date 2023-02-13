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
import org.apache.flink.api.common.typeutils.base.BooleanSerializer;
import org.apache.flink.api.common.typeutils.base.GenericArraySerializer;
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.api.java.typeutils.runtime.kryo.KryoSerializer;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.datastorage.IterationSharedStorage;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LearningNode;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.common.gbt.loss.Loss;
import org.apache.flink.ml.common.gbt.typeinfo.BinnedInstanceSerializer;
import org.apache.flink.ml.common.gbt.typeinfo.LearningNodeSerializer;
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
import org.eclipse.collections.impl.map.mutable.primitive.IntIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;

/**
 * Calculates local histograms for local data partition. Specifically in the first round, this
 * operator caches all data instances to JVM static region.
 */
public class CacheDataCalcLocalHistsOperator extends AbstractStreamOperator<Histogram>
        implements TwoInputStreamOperator<Row, TrainContext, Histogram>,
                IterationListener<Histogram> {
    private static final Logger LOG =
            LoggerFactory.getLogger(CacheDataCalcLocalHistsOperator.class);

    private static final String TREE_INITIALIZER_STATE_NAME = "tree_initializer";
    private static final String HIST_BUILDER_STATE_NAME = "hist_builder";

    private final GbtParams gbtParams;
    private final IterationID iterationID;

    // States of local data.
    private transient ListStateWithCache<BinnedInstance> instancesCollecting;
    private transient ListState<TreeInitializer> treeInitializer;
    private transient ListState<HistBuilder> histBuilder;

    // Readers/writers of shared data.
    private transient IterationSharedStorage.Writer<BinnedInstance[]> instancesWriter;
    private transient IterationSharedStorage.Reader<PredGradHess[]> pghReader;
    private transient IterationSharedStorage.Writer<int[]> shuffledIndicesWriter;
    private transient IterationSharedStorage.Reader<int[]> swappedIndicesReader;
    private IterationSharedStorage.Writer<int[]> nodeFeaturePairsWriter;
    private IterationSharedStorage.Reader<List<LearningNode>> layerReader;
    private IterationSharedStorage.Writer<LearningNode> rootLearningNodeWriter;
    private IterationSharedStorage.Reader<Boolean> needInitTreeReader;
    private IterationSharedStorage.Writer<Boolean> hasInitedTreeWriter;
    private IterationSharedStorage.Writer<TrainContext> trainContextWriter;

    public CacheDataCalcLocalHistsOperator(GbtParams gbtParams, IterationID iterationID) {
        super();
        this.gbtParams = gbtParams;
        this.iterationID = iterationID;
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
        treeInitializer =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        TREE_INITIALIZER_STATE_NAME, TreeInitializer.class));
        histBuilder =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        HIST_BUILDER_STATE_NAME, HistBuilder.class));

        int subtaskId = getRuntimeContext().getIndexOfThisSubtask();
        instancesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.INSTANCES,
                        getOperatorID(),
                        new GenericArraySerializer<>(
                                BinnedInstance.class, BinnedInstanceSerializer.INSTANCE),
                        new BinnedInstance[0]);
        instancesWriter.initializeState(context);

        shuffledIndicesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.SHUFFLED_INDICES,
                        getOperatorID(),
                        IntPrimitiveArraySerializer.INSTANCE,
                        new int[0]);
        shuffledIndicesWriter.initializeState(context);

        nodeFeaturePairsWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.NODE_FEATURE_PAIRS,
                        getOperatorID(),
                        IntPrimitiveArraySerializer.INSTANCE,
                        new int[0]);
        nodeFeaturePairsWriter.initializeState(context);

        rootLearningNodeWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.ROOT_LEARNING_NODE,
                        getOperatorID(),
                        LearningNodeSerializer.INSTANCE,
                        new LearningNode());
        rootLearningNodeWriter.initializeState(context);

        hasInitedTreeWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.HAS_INITED_TREE,
                        getOperatorID(),
                        BooleanSerializer.INSTANCE,
                        false);
        hasInitedTreeWriter.initializeState(context);

        trainContextWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        SharedKeys.TRAIN_CONTEXT,
                        getOperatorID(),
                        new KryoSerializer<>(TrainContext.class, getExecutionConfig()),
                        new TrainContext());
        trainContextWriter.initializeState(context);

        this.pghReader =
                IterationSharedStorage.getReader(
                        iterationID, subtaskId, SharedKeys.PREDS_GRADS_HESSIANS);
        this.swappedIndicesReader =
                IterationSharedStorage.getReader(
                        iterationID, subtaskId, SharedKeys.SWAPPED_INDICES);
        this.layerReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, SharedKeys.LAYER);
        this.needInitTreeReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, SharedKeys.NEED_INIT_TREE);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        instancesCollecting.snapshotState(context);
        instancesWriter.snapshotState(context);
        shuffledIndicesWriter.snapshotState(context);
        hasInitedTreeWriter.snapshotState(context);
    }

    @Override
    public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
        Row row = streamRecord.getValue();
        IntIntHashMap features;
        if (gbtParams.isInputVector) {
            Vector vec = row.getFieldAs(gbtParams.vectorCol);
            SparseVector sv = vec.toSparse();
            features = new IntIntHashMap(sv.indices.length);
            for (int i = 0; i < sv.indices.length; i += 1) {
                features.put(sv.indices[i], (int) sv.values[i]);
            }
        } else {
            features = new IntIntHashMap(gbtParams.featureCols.length);
            for (int i = 0; i < gbtParams.featureCols.length; i += 1) {
                // Values from StringIndexModel#transform are double.
                features.put(i, ((Number) row.getFieldAs(gbtParams.featureCols[i])).intValue());
            }
        }
        double label = row.getFieldAs(gbtParams.labelCol);
        instancesCollecting.add(new BinnedInstance(features, 1., label));
    }

    @Override
    public void processElement2(StreamRecord<TrainContext> streamRecord) throws Exception {
        TrainContext trainContext = streamRecord.getValue();
        if (null != trainContext) {
            // Not null only in first round.
            trainContextWriter.set(trainContext);
        }
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Histogram> out) throws Exception {
        if (0 == epochWatermark) {
            // Initializes local state in first round.
            instancesWriter.set(
                    (BinnedInstance[])
                            IteratorUtils.toArray(
                                    instancesCollecting.get().iterator(), BinnedInstance.class));
            instancesCollecting.clear();
            TrainContext trainContext =
                    new TrainContextInitializer(gbtParams)
                            .init(
                                    trainContextWriter.get(),
                                    getRuntimeContext().getIndexOfThisSubtask(),
                                    getRuntimeContext().getNumberOfParallelSubtasks(),
                                    instancesWriter.get());
            trainContextWriter.set(trainContext);

            treeInitializer.update(Collections.singletonList(new TreeInitializer(trainContext)));
            histBuilder.update(Collections.singletonList(new HistBuilder(trainContext)));
        }

        TrainContext trainContext = trainContextWriter.get();
        BinnedInstance[] instances = instancesWriter.get();
        Preconditions.checkArgument(
                getRuntimeContext().getIndexOfThisSubtask() == trainContext.subtaskId);
        PredGradHess[] pgh = pghReader.get();

        // In the first round, use prior as the predictions.
        if (0 == pgh.length) {
            pgh = new PredGradHess[instances.length];
            double prior = trainContext.prior;
            Loss loss = trainContext.loss;
            for (int i = 0; i < instances.length; i += 1) {
                double label = instances[i].label;
                pgh[i] =
                        new PredGradHess(
                                prior, loss.gradient(prior, label), loss.hessian(prior, label));
            }
        }

        int[] indices;
        if (needInitTreeReader.get()) {
            TreeInitializer treeInit =
                    OperatorStateUtils.getUniqueElement(
                                    treeInitializer, TREE_INITIALIZER_STATE_NAME)
                            .get();

            // When last tree is finished, initializes a new tree, and shuffle instance indices.
            treeInit.init(shuffledIndicesWriter::set);

            LearningNode rootLearningNode = treeInit.getRootLearningNode();
            indices = shuffledIndicesWriter.get();
            rootLearningNodeWriter.set(rootLearningNode);
            hasInitedTreeWriter.set(true);
        } else {
            // Otherwise, uses the swapped instance indices.
            shuffledIndicesWriter.set(new int[0]);
            indices = swappedIndicesReader.get();
            hasInitedTreeWriter.set(false);
        }

        List<LearningNode> layer = layerReader.get();
        if (layer.size() == 0) {
            layer = Collections.singletonList(rootLearningNodeWriter.get());
        }

        int[] nodeFeaturePairs =
                OperatorStateUtils.getUniqueElement(histBuilder, HIST_BUILDER_STATE_NAME)
                        .get()
                        .getNodeFeaturePairs(layer.size());
        nodeFeaturePairsWriter.set(nodeFeaturePairs);

        Histogram localHists =
                OperatorStateUtils.getUniqueElement(histBuilder, HIST_BUILDER_STATE_NAME)
                        .get()
                        .build(layer, nodeFeaturePairs, indices, instances, pgh);
        out.collect(localHists);
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Histogram> collector) {
        instancesCollecting.clear();
        treeInitializer.clear();
        histBuilder.clear();

        instancesWriter.set(new BinnedInstance[0]);
        shuffledIndicesWriter.set(new int[0]);
        nodeFeaturePairsWriter.set(new int[0]);
    }

    @Override
    public void close() throws Exception {
        instancesWriter.remove();
        shuffledIndicesWriter.remove();
        nodeFeaturePairsWriter.remove();
        rootLearningNodeWriter.remove();
        hasInitedTreeWriter.remove();
        trainContextWriter.remove();
        super.close();
    }
}
