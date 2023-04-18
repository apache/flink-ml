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

    /** Names of the broadcast data streams. */
    private final String[] broadcastStreamNames;

    /** Output types of input data streams. */
    private final TypeInformation<?>[] inTypes;

    /** Input list of the multi-input operator. */
    @SuppressWarnings("rawtypes")
    private final List<Input> inputList;

    /** Whether each broadcast input has finished. */
    private boolean[] cachesReady;

    /** State storage of the broadcast inputs. */
    @SuppressWarnings("rawtypes")
    private ListState[] cacheStates;

    /** CacheReady state storage of the broadcast inputs. */
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
        this.cachesReady = new boolean[inTypes.length];
        this.cacheStates = new ListState[inTypes.length];
        this.cacheReadyStates = new ListState[inTypes.length];
    }

    @Override
    public List<Input> getInputs() {
        return inputList;
    }

    @Override
    @SuppressWarnings({"unchecked"})
    public void endInput(int i) throws Exception {
        cachesReady[i - 1] = true;
        String key =
                broadcastStreamNames[i - 1] + "-" + getRuntimeContext().getIndexOfThisSubtask();
        BroadcastContext.putBroadcastVariable(
                key,
                Tuple2.of(
                        true,
                        IteratorUtils.toList(
                                ((ListState<?>) cacheStates[i - 1]).get().iterator())));
        BroadcastContext.notifyCacheFinished(key);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        for (int i = 0; i < inTypes.length; i++) {
            cacheReadyStates[i].clear();
            cacheReadyStates[i].add(cachesReady[i]);
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
            // TODO: there may be a memory leak if the BroadcastWrapper finishes fast before this
            // task finishes.
            BroadcastContext.putBroadcastVariable(
                    broadcastStreamNames[i] + "-" + getRuntimeContext().getIndexOfThisSubtask(),
                    Tuple2.of(
                            cacheReady,
                            IteratorUtils.toList(
                                    ((ListState<?>) cacheStates[i]).get().iterator())));
        }
    }

    private class ProxyInput<IN, OT> extends AbstractInput<IN, OT> {

        public ProxyInput(AbstractStreamOperatorV2<OT> owner, int inputId) {
            super(owner, inputId);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void processElement(StreamRecord<IN> element) throws Exception {
            (cacheStates[inputId - 1]).add(element.getValue());
        }
    }
}
