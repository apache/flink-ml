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

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.message.Message;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Preconditions;

/**
 * Assembles the responses from different servers for one pull request.
 *
 * <p>Note that for each single-thread worker, there are at exactly #numServers segments for each
 * pull request in the responses.
 */
public class ResponseAssemblerOperator extends AbstractStreamOperator<byte[]>
        implements OneInputStreamOperator<Tuple2<Integer, byte[]>, byte[]> {
    private final int numServers;

    private int workerId;

    private int numResponsesReceived = 0;
    private ListState<Integer> numResponsesReceivedState;

    private ListState<byte[]> responsesReceived;

    public ResponseAssemblerOperator(int numServers) {
        this.numServers = numServers;
    }

    @Override
    public void open() throws Exception {
        super.open();
        this.workerId = getRuntimeContext().getIndexOfThisSubtask();
    }

    @Override
    public void processElement(StreamRecord<Tuple2<Integer, byte[]>> element) throws Exception {
        Preconditions.checkState(element.getValue().f0 == workerId);
        responsesReceived.add(element.getValue().f1);
        numResponsesReceived++;

        if (numResponsesReceived == numServers) {
            Message message = Message.assembleMessages(responsesReceived.get().iterator());
            output.collect(new StreamRecord<>(message.bytes));
            responsesReceived.clear();
            numResponsesReceived = 0;
        }
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        responsesReceived =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "responsesReceivedState",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));
        numResponsesReceivedState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>("numResponsesReceivedState", Types.INT));
        numResponsesReceived =
                OperatorStateUtils.getUniqueElement(
                                numResponsesReceivedState, "numResponsesReceived")
                        .orElse(0);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        responsesReceived.clear();
        if (numResponsesReceived > 0) {
            numResponsesReceivedState.clear();
            numResponsesReceivedState.add(numResponsesReceived);
        }
    }
}
