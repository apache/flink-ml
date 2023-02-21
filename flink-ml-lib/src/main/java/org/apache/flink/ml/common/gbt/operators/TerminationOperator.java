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

import org.apache.flink.iteration.IterationListener;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.common.sharedstorage.SharedStorageContext;
import org.apache.flink.ml.common.sharedstorage.SharedStorageStreamOperator;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import java.util.UUID;

/** Determines whether to terminated training. */
public class TerminationOperator extends AbstractStreamOperator<Integer>
        implements OneInputStreamOperator<Integer, Integer>,
                IterationListener<GBTModelData>,
                SharedStorageStreamOperator {

    private final OutputTag<GBTModelData> modelDataOutputTag;
    private final String sharedStorageAccessorID;
    private transient SharedStorageContext sharedStorageContext;

    public TerminationOperator(OutputTag<GBTModelData> modelDataOutputTag) {
        this.modelDataOutputTag = modelDataOutputTag;
        sharedStorageAccessorID = getClass().getSimpleName() + "-" + UUID.randomUUID();
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        sharedStorageContext.initializeState(this, getRuntimeContext(), context);
    }

    @Override
    public void processElement(StreamRecord<Integer> element) throws Exception {}

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, Context context, Collector<GBTModelData> collector)
            throws Exception {
        sharedStorageContext.invoke(
                (getter, setter) -> {
                    boolean terminated =
                            getter.get(SharedStorageConstants.ALL_TREES).size()
                                    == getter.get(SharedStorageConstants.TRAIN_CONTEXT)
                                            .params
                                            .maxIter;
                    // TODO: add validation error rate
                    if (!terminated) {
                        output.collect(new StreamRecord<>(0));
                    }
                });
    }

    @Override
    public void onIterationTerminated(Context context, Collector<GBTModelData> collector)
            throws Exception {
        sharedStorageContext.invoke(
                (getter, setter) ->
                        context.output(
                                modelDataOutputTag,
                                GBTModelData.from(
                                        getter.get(SharedStorageConstants.TRAIN_CONTEXT),
                                        getter.get(SharedStorageConstants.ALL_TREES))));
    }

    @Override
    public void onSharedStorageContextSet(SharedStorageContext context) {
        sharedStorageContext = context;
    }

    @Override
    public String getSharedStorageAccessorID() {
        return sharedStorageAccessorID;
    }
}
