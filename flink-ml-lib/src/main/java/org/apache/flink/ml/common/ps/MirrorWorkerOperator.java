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
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.ps.message.PulledValueM;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * Merges the message from different servers for one pull request.
 *
 * <p>Note that for each single-thread worker, there are at exactly #numServers segments for each
 * pull request in the feedback edge.
 */
public class MirrorWorkerOperator extends AbstractStreamOperator<byte[]>
        implements OneInputStreamOperator<Tuple2<Integer, byte[]>, byte[]> {
    private final int numServers;
    private int workerId;

    /** The received messages from servers for the current pull request. */
    private List<PulledValueM> messageReceived;

    private ListState<byte[]> messageReceivedState;

    public MirrorWorkerOperator(int numServers) {
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
        PulledValueM pulledValueM = PulledValueM.fromBytes(element.getValue().f1);
        messageReceived.add(pulledValueM);
        trySendingPulls(numServers);
    }

    private void trySendingPulls(int numSegments) {
        if (messageReceived.size() == numSegments) {
            Comparator<PulledValueM> comparator = Comparator.comparingInt(o -> o.serverId);
            messageReceived.sort(comparator);
            int size = 0;
            for (PulledValueM pulledValueM : messageReceived) {
                size += pulledValueM.values.length;
            }
            double[] answer = new double[size];
            int offset = 0;
            for (PulledValueM pulledValueM : messageReceived) {
                double[] values = pulledValueM.values;
                System.arraycopy(values, 0, answer, offset, values.length);
                offset += values.length;
            }
            PulledValueM pulledValueM = new PulledValueM(-1, workerId, answer);
            output.collect(new StreamRecord<>(pulledValueM.toBytes()));
            messageReceived.clear();
        }
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        messageReceivedState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "messageReceivedState",
                                        PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO));
        messageReceived = new ArrayList<>();

        Iterator<byte[]> iterator = messageReceivedState.get().iterator();
        if (iterator.hasNext()) {
            while (iterator.hasNext()) {
                messageReceived.add(PulledValueM.fromBytes(iterator.next()));
            }
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        super.snapshotState(context);
        messageReceivedState.clear();
        if (messageReceived.size() > 0) {
            for (PulledValueM valuesPulled : messageReceived) {
                messageReceivedState.add(valuesPulled.toBytes());
            }
        }
    }
}
