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

import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.ml.common.gbt.datastorage.IterationSharedStorage;
import org.apache.flink.ml.common.gbt.defs.BinnedInstance;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.loss.Loss;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import org.eclipse.collections.impl.map.mutable.primitive.IntIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Calculates local histograms for local data partition. Specifically in the first round, this
 * operator caches all data instances to JVM static region.
 */
public class CacheDataCalcLocalHistsOperator extends AbstractStreamOperator<Histogram>
        implements TwoInputStreamOperator<Row, LocalState, Histogram>,
                IterationListener<Histogram> {
    private static final Logger LOG =
            LoggerFactory.getLogger(CacheDataCalcLocalHistsOperator.class);

    private final GbtParams gbtParams;
    private final IterationID iterationID;
    private final String sharedInstancesKey;
    private final String sharedPredGradHessKey;
    private final String sharedShuffledIndicesKey;
    private final String sharedSwappedIndicesKey;
    private final OutputTag<LocalState> stateOutputTag;

    // States of local data.
    private transient List<BinnedInstance> instancesCollecting;
    private transient LocalState localState;
    private transient TreeInitializer treeInitializer;
    private transient HistBuilder histBuilder;

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
    public void open() throws Exception {
        instancesCollecting = new ArrayList<>();

        int subtaskId = getRuntimeContext().getIndexOfThisSubtask();
        instancesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        sharedInstancesKey,
                        getOperatorID(),
                        new BinnedInstance[0]);

        shuffledIndicesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        sharedShuffledIndicesKey,
                        getOperatorID(),
                        new int[0]);

        this.pghReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, sharedPredGradHessKey);
        this.swappedIndicesReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, sharedSwappedIndicesKey);
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
        localState = streamRecord.getValue();
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<Histogram> out) throws Exception {
        if (0 == epochWatermark) {
            // Initializes local state in first round.
            instancesWriter.set(instancesCollecting.toArray(new BinnedInstance[0]));
            instancesCollecting.clear();
            new LocalStateInitializer(gbtParams)
                    .init(
                            localState,
                            getRuntimeContext().getIndexOfThisSubtask(),
                            getRuntimeContext().getNumberOfParallelSubtasks(),
                            instancesWriter.get());

            treeInitializer = new TreeInitializer(localState.statics);
            histBuilder = new HistBuilder(localState.statics);
        }

        BinnedInstance[] instances = instancesWriter.get();
        Preconditions.checkArgument(
                getRuntimeContext().getIndexOfThisSubtask() == localState.statics.subtaskId);
        PredGradHess[] pgh = pghReader.get();

        // In the first round, use prior as the predictions.
        if (0 == pgh.length) {
            pgh = new PredGradHess[instances.length];
            double prior = localState.statics.prior;
            Loss loss = localState.statics.loss;
            for (int i = 0; i < instances.length; i += 1) {
                double label = instances[i].label;
                pgh[i] =
                        new PredGradHess(
                                prior, loss.gradient(prior, label), loss.hessian(prior, label));
            }
        }

        int[] indices;
        if (!localState.dynamics.inWeakLearner) {
            // When last tree is finished, initializes a new tree, and shuffle instance indices.
            treeInitializer.init(localState.dynamics, shuffledIndicesWriter::set);
            localState.dynamics.inWeakLearner = true;
            indices = shuffledIndicesWriter.get();
        } else {
            // Otherwise, uses the swapped instance indices.
            shuffledIndicesWriter.set(new int[0]);
            indices = swappedIndicesReader.get();
        }

        Histogram localHists =
                histBuilder.build(
                        localState.dynamics.layer,
                        localState.dynamics.nodeFeaturePairs,
                        indices,
                        instances,
                        pgh);
        out.collect(localHists);
        context.output(stateOutputTag, localState);
    }

    @Override
    public void onIterationTerminated(Context context, Collector<Histogram> collector) {
        instancesCollecting.clear();
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
