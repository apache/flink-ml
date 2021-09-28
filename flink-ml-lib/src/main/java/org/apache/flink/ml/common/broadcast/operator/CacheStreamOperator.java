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

package org.apache.flink.ml.common.broadcast.operator;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.broadcast.BroadcastContext;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractInput;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorV2;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.apache.commons.collections.IteratorUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/** The operator that process all broadcast inputs and stores them in {@link BroadcastContext}. */
public class CacheStreamOperator<OUT> extends AbstractStreamOperatorV2<OUT>
        implements MultipleInputStreamOperator<OUT>, BoundedMultiInput, Serializable {
    /** names of the broadcast DataStreams. */
    private final String[] broadcastNames;
    /** input list of the multi-input operator. */
    private final List<Input> inputList;
    /** output types of input DataStreams. */
    private final TypeInformation<?>[] inTypes;
    /** caches of the broadcast inputs. */
    private final List<?>[] caches;
    /** state storage of the broadcast inputs. */
    private ListState<?>[] cacheStates;
    /** cacheReady state storage of the broadcast inputs. */
    private ListState<Boolean>[] cacheReadyStates;

    public CacheStreamOperator(
            StreamOperatorParameters<OUT> parameters,
            String[] broadcastNames,
            TypeInformation<?>[] inTypes) {
        super(parameters, broadcastNames.length);
        this.broadcastNames = broadcastNames;
        this.inTypes = inTypes;
        this.caches = new List[inTypes.length];
        for (int i = 0; i < inTypes.length; i++) {
            caches[i] = new ArrayList<>();
        }
        this.cacheStates = new ListState[inTypes.length];
        this.cacheReadyStates = new ListState[inTypes.length];

        inputList = new ArrayList<>();
        for (int i = 0; i < inTypes.length; i++) {
            inputList.add(new ProxyInput(this, i + 1));
        }
    }

    @Override
    public List<Input> getInputs() {
        return inputList;
    }

    @Override
    public void endInput(int i) {
        BroadcastContext.markCacheFinished(
                Tuple2.of(broadcastNames[i - 1], getRuntimeContext().getIndexOfThisSubtask()));
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        for (int i = 0; i < inTypes.length; i++) {
            cacheStates[i].clear();
            cacheStates[i].addAll((List) caches[i]);
            cacheReadyStates[i].clear();
            boolean isCacheFinished =
                    BroadcastContext.isCacheFinished(
                            Tuple2.of(
                                    broadcastNames[i],
                                    getRuntimeContext().getIndexOfThisSubtask()));
            cacheReadyStates[i].add(isCacheFinished);
        }
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        for (int i = 0; i < inTypes.length; i++) {
            cacheStates[i] =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor("cache_data_" + i, inTypes[i]));
            caches[i] = IteratorUtils.toList(cacheStates[i].get().iterator());

            cacheReadyStates[i] =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<Boolean>(
                                            "cache_ready_state_" + i,
                                            BasicTypeInfo.BOOLEAN_TYPE_INFO));
            List<Boolean> restoredCacheReady =
                    IteratorUtils.toList(cacheReadyStates[i].get().iterator());
            boolean cacheReady = restoredCacheReady.size() == 0 ? false : restoredCacheReady.get(0);
            BroadcastContext.putBroadcastVariable(
                    Tuple2.of(broadcastNames[i], getRuntimeContext().getIndexOfThisSubtask()),
                    Tuple2.of(cacheReady, caches[i]));
        }
    }

    private class ProxyInput<IN, OUT> extends AbstractInput<IN, OUT> {

        public ProxyInput(AbstractStreamOperatorV2<OUT> owner, int inputId) {
            super(owner, inputId);
        }

        @Override
        public void processElement(StreamRecord<IN> element) {
            ((List<IN>) caches[inputId - 1]).add(element.getValue());
        }
    }
}
