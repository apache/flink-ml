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
import org.apache.flink.ml.common.gbt.defs.LocalState;
import org.apache.flink.ml.common.gbt.defs.PredGradHess;
import org.apache.flink.ml.common.gbt.defs.Splits;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

/**
 * Post-process after global splits obtained, including split instances to left or child nodes, and
 * update instances scores after a tree is complete.
 */
public class PostSplitsOperator extends AbstractStreamOperator<LocalState>
        implements TwoInputStreamOperator<LocalState, Splits, LocalState>,
                IterationListener<LocalState> {

    private final IterationID iterationID;
    private final String sharedInstancesKey;
    private final String sharedPredGradHessKey;
    private final String sharedShuffledIndicesKey;
    private final String sharedSwappedIndicesKey;
    private final OutputTag<LocalState> finalStateOutputTag;

    private IterationSharedStorage.Reader<BinnedInstance[]> instancesReader;
    private IterationSharedStorage.Writer<PredGradHess[]> pghWriter;
    private IterationSharedStorage.Reader<int[]> shuffledIndicesReader;
    private IterationSharedStorage.Writer<int[]> swappedIndicesWriter;

    private transient LocalState localState;
    private transient Splits splits;
    private transient NodeSplitter nodeSplitter;
    private transient InstanceUpdater instanceUpdater;

    public PostSplitsOperator(
            IterationID iterationID,
            String sharedInstancesKey,
            String sharedPredGradHessKey,
            String sharedShuffledIndicesKey,
            String sharedSwappedIndicesKey,
            OutputTag<LocalState> finalStateOutputTag) {
        this.iterationID = iterationID;
        this.sharedInstancesKey = sharedInstancesKey;
        this.sharedPredGradHessKey = sharedPredGradHessKey;
        this.sharedShuffledIndicesKey = sharedShuffledIndicesKey;
        this.sharedSwappedIndicesKey = sharedSwappedIndicesKey;
        this.finalStateOutputTag = finalStateOutputTag;
    }

    @Override
    public void open() throws Exception {
        int subtaskId = getRuntimeContext().getIndexOfThisSubtask();
        pghWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        sharedPredGradHessKey,
                        getOperatorID(),
                        new PredGradHess[0]);
        swappedIndicesWriter =
                IterationSharedStorage.getWriter(
                        iterationID,
                        subtaskId,
                        sharedSwappedIndicesKey,
                        getOperatorID(),
                        new int[0]);

        this.instancesReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, sharedInstancesKey);
        this.shuffledIndicesReader =
                IterationSharedStorage.getReader(iterationID, subtaskId, sharedShuffledIndicesKey);
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<LocalState> collector) throws Exception {
        LocalState localStateValue = localState;
        if (0 == epochWatermark) {
            nodeSplitter = new NodeSplitter(localStateValue.statics);
            instanceUpdater = new InstanceUpdater(localStateValue.statics);
        }

        int[] indices = swappedIndicesWriter.get();
        if (0 == indices.length) {
            indices = shuffledIndicesReader.get().clone();
        }

        BinnedInstance[] instances = instancesReader.get();
        nodeSplitter.split(
                localStateValue.dynamics.layer,
                localStateValue.dynamics.leaves,
                splits.splits,
                indices,
                instances);

        if (localStateValue.dynamics.layer.isEmpty()) {
            localStateValue.dynamics.inWeakLearner = false;
            instanceUpdater.update(
                    localStateValue.dynamics.leaves, indices, instances, pghWriter::set);
            swappedIndicesWriter.set(new int[0]);
        } else {
            swappedIndicesWriter.set(indices);
        }
        collector.collect(localStateValue);
    }

    @Override
    public void onIterationTerminated(Context context, Collector<LocalState> collector) {
        pghWriter.set(new PredGradHess[0]);
        swappedIndicesWriter.set(new int[0]);
        if (0 == getRuntimeContext().getIndexOfThisSubtask()) {
            context.output(finalStateOutputTag, localState);
        }
    }

    @Override
    public void processElement1(StreamRecord<LocalState> element) throws Exception {
        localState = element.getValue();
    }

    @Override
    public void processElement2(StreamRecord<Splits> element) throws Exception {
        splits = element.getValue();
    }

    @Override
    public void close() throws Exception {
        pghWriter.remove();
        swappedIndicesWriter.remove();
        super.close();
    }
}
