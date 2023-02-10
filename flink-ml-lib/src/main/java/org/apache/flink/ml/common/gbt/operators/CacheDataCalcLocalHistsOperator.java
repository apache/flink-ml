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
import org.apache.flink.api.common.typeutils.base.GenericArraySerializer;
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.gbt.datastorage.IterationSharedStorage;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.loss.Loss;
import org.apache.flink.ml.common.gbt.typeinfo.BinnedInstanceSerializer;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;
import org.eclipse.collections.impl.map.mutable.primitive.IntIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;

/**
 * Calculates local histograms for local data partition. Specifically in the first round, this
 * operator caches all data instances to JVM static region.
 */
public class CacheDataCalcLocalHistsOperator extends AbstractStreamOperator<Histogram>
        implements TwoInputStreamOperator<Row, LocalState, Histogram>,
                IterationListener<Histogram> {
    private static final Logger LOG =
            LoggerFactory.getLogger(CacheDataCalcLocalHistsOperator.class);

    private static final String LOCAL_STATE_STATE_NAME = "local_state";
    private static final String TREE_INITIALIZER_STATE_NAME = "tree_initializer";
    private static final String HIST_BUILDER_STATE_NAME = "hist_builder";

    private final GbtParams gbtParams;
    private final IterationID iterationID;
    private final String sharedInstancesKey;
    private final String sharedPredGradHessKey;
    private final String sharedShuffledIndicesKey;
    private final String sharedSwappedIndicesKey;
    private final OutputTag<LocalState> stateOutputTag;

    // States of local data.
    private transient ListStateWithCache<BinnedInstance> instancesCollecting;
    private transient ListState<LocalState> localState;
    private transient ListState<TreeInitializer> treeInitializer;
    private transient ListState<HistBuilder> histBuilder;

    // Readers/writers of shared data.
    private transient IterationSharedStorage.Writer<BinnedInstance[]> instancesWriter;
    private transient IterationSharedStorage.Reader<PredGradHess[]> pghReader;
    private transient IterationSharedStorage.Writer<int[]> shuffledIndicesWriter;
    private transient IterationSharedStorage.Reader<int[]> swappedIndicesReader;

    public CacheDataCalcLocalHistsOperator(
            GbtParams gbtParams,
            IterationID iterationID,
            String sharedInstancesKey,
            String sharedPredGradHessKey,
            String sharedShuffledIndicesKey,
            String sharedSwappedIndicesKey,
            OutputTag<LocalState> stateOutputTag) {
        super();
        this.gbtParams = gbtParams;
        this.iterationID = iterationID;
        this.sharedInstancesKey = sharedInstancesKey;
        this.sharedPredGradHessKey = sharedPredGradHessKey;
        this.sharedShuffledIndicesKey = sharedShuffledIndicesKey;
        this.sharedSwappedIndicesKey = sharedSwappedIndicesKey;
        this.stateOutputTag = stateOutputTag;
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
        localState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        LOCAL_STATE_STATE_NAME, LocalState.class));
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
                        sharedInstancesKey,
                        getOperatorID(),
                        new GenericArraySerializer<>(
                                BinnedInstance.class, BinnedInstanceSerializer.INSTANCE),
                        new BinnedInstance[0]);
        instancesWriter.initializeState(context);

        shuffledIndicesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        sharedShuffledIndicesKey,
                        getOperatorID(),
                        IntPrimitiveArraySerializer.INSTANCE,
                        new int[0]);
        shuffledIndicesWriter.initializeState(context);

        this.pghReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, sharedPredGradHessKey);
        this.swappedIndicesReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, sharedSwappedIndicesKey);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        instancesCollecting.snapshotState(context);
        instancesWriter.snapshotState(context);
        shuffledIndicesWriter.snapshotState(context);
    }

    @Override
    public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
        Row row = streamRecord.getValue();
        IntIntHashMap features = new IntIntHashMap();
        if (gbtParams.isInputVector) {
            Vector vec = row.getFieldAs(gbtParams.vectorCol);
            SparseVector sv = vec.toSparse();
            for (int i = 0; i < sv.indices.length; i += 1) {
                features.put(sv.indices[i], (int) sv.values[i]);
            }
        } else {
            for (int i = 0; i < gbtParams.featureCols.length; i += 1) {
                // Values from StringIndexModel#transform are double.
                features.put(i, ((Number) row.getFieldAs(gbtParams.featureCols[i])).intValue());
            }
        }
        double label = row.getFieldAs(gbtParams.labelCol);
        instancesCollecting.add(new BinnedInstance(features, 1., label));
    }

    @Override
    public void processElement2(StreamRecord<LocalState> streamRecord) throws Exception {
        localState.update(Collections.singletonList(streamRecord.getValue()));
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Histogram> out) throws Exception {
        LocalState localStateValue =
                OperatorStateUtils.getUniqueElement(localState, "local_state").get();
        if (0 == epochWatermark) {
            // Initializes local state in first round.
            instancesWriter.set(
                    (BinnedInstance[])
                            IteratorUtils.toArray(
                                    instancesCollecting.get().iterator(), BinnedInstance.class));
            instancesCollecting.clear();
            new LocalStateInitializer(gbtParams)
                    .init(
                            localStateValue,
                            getRuntimeContext().getIndexOfThisSubtask(),
                            getRuntimeContext().getNumberOfParallelSubtasks(),
                            instancesWriter.get());

            treeInitializer.update(
                    Collections.singletonList(new TreeInitializer(localStateValue.statics)));
            histBuilder.update(Collections.singletonList(new HistBuilder(localStateValue.statics)));
        }

        BinnedInstance[] instances = instancesWriter.get();
        Preconditions.checkArgument(
                getRuntimeContext().getIndexOfThisSubtask() == localStateValue.statics.subtaskId);
        PredGradHess[] pgh = pghReader.get();

        // In the first round, use prior as the predictions.
        if (0 == pgh.length) {
            pgh = new PredGradHess[instances.length];
            double prior = localStateValue.statics.prior;
            Loss loss = localStateValue.statics.loss;
            for (int i = 0; i < instances.length; i += 1) {
                double label = instances[i].label;
                pgh[i] =
                        new PredGradHess(
                                prior, loss.gradient(prior, label), loss.hessian(prior, label));
            }
        }

        int[] indices;
        if (!localStateValue.dynamics.inWeakLearner) {
            // When last tree is finished, initializes a new tree, and shuffle instance indices.
            OperatorStateUtils.getUniqueElement(treeInitializer, TREE_INITIALIZER_STATE_NAME)
                    .get()
                    .init(localStateValue.dynamics, shuffledIndicesWriter::set);
            localStateValue.dynamics.inWeakLearner = true;
            indices = shuffledIndicesWriter.get();
        } else {
            // Otherwise, uses the swapped instance indices.
            shuffledIndicesWriter.set(new int[0]);
            indices = swappedIndicesReader.get();
        }

        Histogram localHists =
                OperatorStateUtils.getUniqueElement(histBuilder, HIST_BUILDER_STATE_NAME)
                        .get()
                        .build(
                                localStateValue.dynamics.layer,
                                localStateValue.dynamics.nodeFeaturePairs,
                                indices,
                                instances,
                                pgh);
        out.collect(localHists);
        context.output(stateOutputTag, localStateValue);
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Histogram> collector) {
        instancesCollecting.clear();
        localState.clear();
        treeInitializer.clear();
        histBuilder.clear();

        instancesWriter.set(new BinnedInstance[0]);
        shuffledIndicesWriter.set(new int[0]);
    }

    @Override
    public void close() throws Exception {
        instancesWriter.remove();
        shuffledIndicesWriter.remove();
        super.close();
    }
}
