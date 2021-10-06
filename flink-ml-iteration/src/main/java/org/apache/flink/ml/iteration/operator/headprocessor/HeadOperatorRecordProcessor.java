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

package org.apache.flink.ml.iteration.operator.headprocessor;

import org.apache.flink.api.common.TaskInfo;
import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.operator.HeadOperator;
import org.apache.flink.ml.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.OutputTag;

import java.io.IOException;

/** The component to actually deal with the event received in the {@link HeadOperator}. */
public interface HeadOperatorRecordProcessor {

    void initializeState(HeadOperatorState headOperatorState) throws Exception;

    void processElement(StreamRecord<IterationRecord<?>> record);

    boolean processFeedbackElement(StreamRecord<IterationRecord<?>> record);

    boolean onGloballyAligned(GloballyAlignedEvent globallyAlignedEvent);

    HeadOperatorState snapshotState();

    interface Context {

        StreamConfig getStreamConfig();

        TaskInfo getTaskInfo();

        void output(StreamRecord<IterationRecord<?>> record);

        void output(
                OutputTag<IterationRecord<?>> outputTag, StreamRecord<IterationRecord<?>> record);

        void broadcastOutput(StreamRecord<IterationRecord<?>> record);

        void updateEpochToCoordinator(int epoch, long numFeedbackRecords);
    }
}
