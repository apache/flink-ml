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

package org.apache.flink.iteration.operator.headprocessor;

import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.runtime.state.StatePartitionStreamProvider;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.FlinkRuntimeException;

/**
 * Processor used after we received terminated globally aligned event from the coordinator, but
 * before we received the (Integer.MAX_VALUE + 1) from the feedback channel again.
 */
public class TerminatingHeadOperatorRecordProcessor implements HeadOperatorRecordProcessor {

    @Override
    public void initializeState(
            HeadOperatorState headOperatorState,
            Iterable<StatePartitionStreamProvider> rawStates) {}

    @Override
    public void processElement(StreamRecord<IterationRecord<?>> record) {
        throw new FlinkRuntimeException(
                "It is not possible to receive the element from normal input during terminating.");
    }

    @Override
    public boolean processFeedbackElement(StreamRecord<IterationRecord<?>> record) {
        if (record.getValue().getType() == IterationRecord.Type.EPOCH_WATERMARK) {
            return record.getValue().getEpoch() == Integer.MAX_VALUE + 1;
        }

        return false;
    }

    @Override
    public boolean onGloballyAligned(GloballyAlignedEvent globallyAlignedEvent) {
        throw new FlinkRuntimeException(
                "It is not possible to receive the globally aligned event from normal input during terminating.");
    }

    @Override
    public HeadOperatorState snapshotState() {
        return null;
    }
}
