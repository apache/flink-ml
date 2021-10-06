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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.operator.HeadOperator;
import org.apache.flink.ml.iteration.operator.OperatorUtils;
import org.apache.flink.ml.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.OutputTag;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Processes the event before we received the terminated global aligned event from the coordinator.
 */
public class RegularHeadOperatorRecordProcessor implements HeadOperatorRecordProcessor {

    protected static final Logger LOG =
            LoggerFactory.getLogger(RegularHeadOperatorRecordProcessor.class);

    private final Context headOperatorContext;

    private final StreamRecord<IterationRecord<?>> reusable;

    private final Map<Integer, Long> numFeedbackRecordsPerRound;

    private final String senderId;

    public RegularHeadOperatorRecordProcessor(Context headOperatorContext) {
        this.headOperatorContext = headOperatorContext;

        this.reusable = new StreamRecord<>(null);
        this.numFeedbackRecordsPerRound = new HashMap<>();

        this.senderId =
                OperatorUtils.getUniqueSenderId(
                        headOperatorContext.getStreamConfig().getOperatorID(),
                        headOperatorContext.getTaskInfo().getIndexOfThisSubtask());
    }

    @Override
    public void initializeState(HeadOperatorState headOperatorState) throws Exception {}

    @Override
    public void processElement(StreamRecord<IterationRecord<?>> element) {
        processRecord(element);
    }

    @Override
    public boolean processFeedbackElement(StreamRecord<IterationRecord<?>> element) {
        if (element.getValue().getType() == IterationRecord.Type.RECORD) {
            numFeedbackRecordsPerRound.compute(
                    element.getValue().getRound(), (round, count) -> count == null ? 1 : count + 1);
        }

        processRecord(element);

        return false;
    }

    @Override
    public boolean onGloballyAligned(GloballyAlignedEvent globallyAlignedEvent) {
        LOG.info("Received global event {}", globallyAlignedEvent);

        reusable.replace(
                IterationRecord.newEpochWatermark(
                        globallyAlignedEvent.isTerminated()
                                ? Integer.MAX_VALUE
                                : globallyAlignedEvent.getRound(),
                        senderId),
                0);
        headOperatorContext.broadcastOutput(reusable);

        // Also notify the listener
        headOperatorContext.output((OutputTag) HeadOperator.ALIGN_NOTIFY_OUTPUT_TAG, reusable);

        return globallyAlignedEvent.isTerminated();
    }

    @Override
    public HeadOperatorState snapshotState() {
        return new HeadOperatorState();
    }

    @VisibleForTesting
    public Map<Integer, Long> getNumFeedbackRecordsPerRound() {
        return numFeedbackRecordsPerRound;
    }

    private void processRecord(StreamRecord<IterationRecord<?>> iterationRecord) {
        switch (iterationRecord.getValue().getType()) {
            case RECORD:
                reusable.replace(iterationRecord.getValue(), iterationRecord.getTimestamp());
                headOperatorContext.output(reusable);
                break;
            case EPOCH_WATERMARK:
                LOG.info("Head Received epoch watermark {}", iterationRecord.getValue().getRound());
                headOperatorContext.updateEpochToCoordinator(
                        iterationRecord.getValue().getRound(),
                        numFeedbackRecordsPerRound.getOrDefault(
                                iterationRecord.getValue().getRound(), 0L));
                break;
        }
    }
}
