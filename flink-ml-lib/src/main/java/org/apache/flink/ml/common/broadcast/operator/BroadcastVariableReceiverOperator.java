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
import org.apache.flink.iteration.operator.OperatorStateUtils;
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
public class BroadcastVariableReceiverOperator<OUT> extends AbstractStreamOperatorV2<OUT>
        implements MultipleInputStreamOperator<OUT>, BoundedMultiInput, Serializable {

    /** names of the broadcast data streams. */
    private final String[] broadcastStreamNames;

    /** output types of input data streams. */
    private final TypeInformation<?>[] inTypes;

    /** input list of the multi-input operator. */
    @SuppressWarnings("rawtypes")
    private final List<Input> inputList;

    /** caches of the broadcast inputs. */
    @SuppressWarnings("rawtypes")
    private final List[] caches;

    /** state storage of the broadcast inputs. */
    private ListState<?>[] cacheStates;

    /** cacheReady state storage of the broadcast inputs. */
    private ListState<Boolean>[] cacheReadyStates;

    @SuppressWarnings({"rawtypes", "unchecked"})
    BroadcastVariableReceiverOperator(
            StreamOperatorParameters<OUT> parameters,
            String[] broadcastStreamNames,
            TypeInformation<?>[] inTypes) {
        super(parameters, broadcastStreamNames.length);
        this.broadcastStreamNames = broadcastStreamNames;
        this.inTypes = inTypes;
        inputList = new ArrayList<>();
        for (int i = 0; i < inTypes.length; i++) {
            inputList.add(new ProxyInput(this, i + 1));
        }
        this.caches = new List[inTypes.length];
        for (int i = 0; i < inTypes.length; i++) {
            caches[i] = new ArrayList<>();
        }
        this.cacheStates = new ListState[inTypes.length];
        this.cacheReadyStates = new ListState[inTypes.length];
    }

    @Override
    public List<Input> getInputs() {
        return inputList;
    }

    @Override
    public void endInput(int i) {
        BroadcastContext.markCacheFinished(
                broadcastStreamNames[i - 1] + "-" + getRuntimeContext().getIndexOfThisSubtask());
    }

    @Override
    @SuppressWarnings("unchecked")
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        for (int i = 0; i < inTypes.length; i++) {
            cacheStates[i].clear();
            cacheStates[i].addAll(caches[i]);
            cacheReadyStates[i].clear();
            boolean isCacheFinished =
                    BroadcastContext.isCacheFinished(
                            broadcastStreamNames[i]
                                    + "-"
                                    + getRuntimeContext().getIndexOfThisSubtask());
            cacheReadyStates[i].add(isCacheFinished);
        }
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
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
                                    new ListStateDescriptor<>(
                                            "cache_ready_state_" + i,
                                            BasicTypeInfo.BOOLEAN_TYPE_INFO));
            boolean cacheReady =
                    OperatorStateUtils.getUniqueElement(
                                    cacheReadyStates[i], "cache_ready_state_" + i)
                            .orElse(false);
            BroadcastContext.putBroadcastVariable(
                    broadcastStreamNames[i] + "-" + getRuntimeContext().getIndexOfThisSubtask(),
                    Tuple2.of(cacheReady, caches[i]));
        }
    }

    private class ProxyInput<IN, OT> extends AbstractInput<IN, OT> {

        public ProxyInput(AbstractStreamOperatorV2<OT> owner, int inputId) {
            super(owner, inputId);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void processElement(StreamRecord<IN> element) {
            (caches[inputId - 1]).add(element.getValue());
        }
    }
}
