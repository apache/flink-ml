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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.runtime.state.StatePartitionStreamProvider;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

import static org.apache.flink.util.Preconditions.checkArgument;
import static org.apache.flink.util.Preconditions.checkState;

/**
 * Processes the event before we received the terminated global aligned event from the coordinator.
 */
public class RegularHeadOperatorRecordProcessor implements HeadOperatorRecordProcessor {

    protected static final Logger LOG =
            LoggerFactory.getLogger(RegularHeadOperatorRecordProcessor.class);

    private final Context headOperatorContext;

    private final Map<Integer, Long> numFeedbackRecordsPerEpoch;

    private final String senderId;

    private int latestRoundAligned;

    private int latestRoundGloballyAligned;

    public RegularHeadOperatorRecordProcessor(Context headOperatorContext) {
        this.headOperatorContext = headOperatorContext;

        this.numFeedbackRecordsPerEpoch = new HashMap<>();

        this.senderId =
                OperatorUtils.getUniqueSenderId(
                        headOperatorContext.getStreamConfig().getOperatorID(),
                        headOperatorContext.getTaskInfo().getIndexOfThisSubtask());

        this.latestRoundAligned = -1;
        this.latestRoundGloballyAligned = -1;
    }

    @Override
    public void initializeState(
            HeadOperatorState headOperatorState, Iterable<StatePartitionStreamProvider> rawStates) {
        checkArgument(headOperatorState != null, "The initialized state should not be null");

        numFeedbackRecordsPerEpoch.putAll(headOperatorState.getNumFeedbackRecordsEachRound());
        latestRoundAligned = headOperatorState.getLatestRoundAligned();
        latestRoundGloballyAligned = headOperatorState.getLatestRoundGloballyAligned();

        // If the only round not fully aligned is round 0, then wait till endOfInput in
        // case the input is changed.
        if (!(latestRoundAligned == 0 && latestRoundGloballyAligned == -1)) {
            for (int i = latestRoundGloballyAligned + 1; i <= latestRoundAligned; ++i) {
                headOperatorContext.updateEpochToCoordinator(
                        i, numFeedbackRecordsPerEpoch.getOrDefault(i, 0L));
            }
        }
    }

    @Override
    public void processElement(StreamRecord<IterationRecord<?>> element) {
        processRecord(element);
    }

    @Override
    public boolean processFeedbackElement(StreamRecord<IterationRecord<?>> element) {
        if (element.getValue().getType() == IterationRecord.Type.RECORD) {
            numFeedbackRecordsPerEpoch.compute(
                    element.getValue().getEpoch(), (epoch, count) -> count == null ? 1 : count + 1);
        }

        processRecord(element);

        return false;
    }

    @Override
    public boolean onGloballyAligned(GloballyAlignedEvent globallyAlignedEvent) {
        LOG.info("Received global event {}", globallyAlignedEvent);
        checkState(
                (globallyAlignedEvent.getEpoch() == 0 && latestRoundGloballyAligned == 0)
                        || globallyAlignedEvent.getEpoch() > latestRoundGloballyAligned,
                String.format(
                        "Receive unexpected global aligned event, latest = %d, this one = %d",
                        latestRoundGloballyAligned, globallyAlignedEvent.getEpoch()));

        StreamRecord<IterationRecord<?>> record =
                new StreamRecord<>(
                        IterationRecord.newEpochWatermark(
                                globallyAlignedEvent.isTerminated()
                                        ? Integer.MAX_VALUE
                                        : globallyAlignedEvent.getEpoch(),
                                senderId),
                        0);
        headOperatorContext.broadcastOutput(record);

        latestRoundGloballyAligned =
                Math.max(globallyAlignedEvent.getEpoch(), latestRoundGloballyAligned);
        return globallyAlignedEvent.isTerminated();
    }

    @Override
    public HeadOperatorState snapshotState() {
        return new HeadOperatorState(
                new HashMap<>(numFeedbackRecordsPerEpoch),
                latestRoundAligned,
                latestRoundGloballyAligned);
    }

    @VisibleForTesting
    public Map<Integer, Long> getNumFeedbackRecordsPerEpoch() {
        return numFeedbackRecordsPerEpoch;
    }

    @VisibleForTesting
    public int getLatestRoundAligned() {
        return latestRoundAligned;
    }

    @VisibleForTesting
    public int getLatestRoundGloballyAligned() {
        return latestRoundGloballyAligned;
    }

    private void processRecord(StreamRecord<IterationRecord<?>> iterationRecord) {
        switch (iterationRecord.getValue().getType()) {
            case RECORD:
                headOperatorContext.output(iterationRecord);
                break;
            case EPOCH_WATERMARK:
                LOG.info("Head Received epoch watermark {}", iterationRecord.getValue().getEpoch());

                boolean needNotifyCoordinator = false;
                if (iterationRecord.getValue().getEpoch() == 0) {
                    if (latestRoundAligned <= 0) {
                        needNotifyCoordinator = true;
                    }
                } else {
                    checkState(
                            iterationRecord.getValue().getEpoch() > latestRoundAligned,
                            String.format(
                                    "Unexpected epoch watermark: latest = %d, this one = %d",
                                    latestRoundAligned, iterationRecord.getValue().getEpoch()));
                    headOperatorContext.updateEpochToCoordinator(
                            iterationRecord.getValue().getEpoch(),
                            numFeedbackRecordsPerEpoch.getOrDefault(
                                    iterationRecord.getValue().getEpoch(), 0L));
                }

                if (needNotifyCoordinator) {
                    headOperatorContext.updateEpochToCoordinator(
                            iterationRecord.getValue().getEpoch(),
                            numFeedbackRecordsPerEpoch.getOrDefault(
                                    iterationRecord.getValue().getEpoch(), 0L));
                }

                latestRoundAligned =
                        Math.max(iterationRecord.getValue().getEpoch(), latestRoundAligned);
                break;
        }
    }
}
