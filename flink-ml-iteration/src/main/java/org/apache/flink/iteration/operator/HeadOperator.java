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

package org.apache.flink.iteration.operator;

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.operators.MailboxExecutor;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.broadcast.BroadcastOutput;
import org.apache.flink.iteration.broadcast.BroadcastOutputFactory;
import org.apache.flink.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.iteration.operator.event.SubtaskAlignedEvent;
import org.apache.flink.runtime.operators.coordination.OperatorEvent;
import org.apache.flink.runtime.operators.coordination.OperatorEventGateway;
import org.apache.flink.runtime.operators.coordination.OperatorEventHandler;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannel;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannelBroker;
import org.apache.flink.statefun.flink.core.feedback.FeedbackConsumer;
import org.apache.flink.statefun.flink.core.feedback.FeedbackKey;
import org.apache.flink.statefun.flink.core.feedback.SubtaskFeedbackKey;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.ProcessingTimeService;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.ExceptionUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executor;

/**
 * The head operators unions the initialized variable stream and the feedback stream, and
 * synchronize the epoch watermark (round).
 */
public class HeadOperator extends AbstractStreamOperator<IterationRecord<?>>
        implements OneInputStreamOperator<IterationRecord<?>, IterationRecord<?>>,
                FeedbackConsumer<StreamRecord<IterationRecord<?>>>,
                OperatorEventHandler,
                BoundedOneInput {

    private final IterationID iterationId;

    private final int feedbackIndex;

    private final boolean isCriteriaStream;

    private final OperatorEventGateway operatorEventGateway;

    private final MailboxExecutor mailboxExecutor;

    private final Map<Integer, Long> numFeedbackRecordsPerEpoch;

    private transient String uniqueSenderId;

    private transient BroadcastOutput<?> eventBroadcastOutput;

    private transient StreamRecord<IterationRecord<?>> reusable;

    private transient boolean shouldTerminate;

    public HeadOperator(
            IterationID iterationId,
            int feedbackIndex,
            boolean isCriteriaStream,
            MailboxExecutor mailboxExecutor,
            OperatorEventGateway operatorEventGateway,
            ProcessingTimeService processingTimeService) {
        this.iterationId = Objects.requireNonNull(iterationId);
        this.feedbackIndex = feedbackIndex;
        this.isCriteriaStream = isCriteriaStream;
        this.mailboxExecutor = Objects.requireNonNull(mailboxExecutor);
        this.operatorEventGateway = Objects.requireNonNull(operatorEventGateway);
        this.numFeedbackRecordsPerEpoch = new HashMap<>();

        // Even though this operator does not use the processing
        // time service, AbstractStreamOperator requires this
        // field is non-null, otherwise we get a NullPointerException
        super.processingTimeService = processingTimeService;
    }

    @Override
    public void setup(
            StreamTask<?, ?> containingTask,
            StreamConfig config,
            Output<StreamRecord<IterationRecord<?>>> output) {
        super.setup(containingTask, config, output);
        uniqueSenderId =
                OperatorUtils.getUniqueSenderId(
                        getOperatorID(), getRuntimeContext().getIndexOfThisSubtask());
        eventBroadcastOutput =
                BroadcastOutputFactory.createBroadcastOutput(
                        output, metrics.getIOMetricGroup().getNumRecordsOutCounter());
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);

        reusable = new StreamRecord<>(null);

        // Here we register a mail
        registerFeedbackConsumer(
                (Runnable runnable) -> {
                    if (!shouldTerminate) {
                        mailboxExecutor.execute(runnable::run, "Head feedback");
                    }
                });
    }

    @Override
    public void processElement(StreamRecord<IterationRecord<?>> element) throws Exception {
        processRecord(element);
    }

    @Override
    public void processFeedback(StreamRecord<IterationRecord<?>> iterationRecord) throws Exception {
        if (iterationRecord.getValue().getType() == IterationRecord.Type.RECORD) {
            numFeedbackRecordsPerEpoch.compute(
                    iterationRecord.getValue().getEpoch(),
                    (round, count) -> count == null ? 1 : count + 1);
        }
        processRecord(iterationRecord);
    }

    private void processRecord(StreamRecord<IterationRecord<?>> iterationRecord) {
        switch (iterationRecord.getValue().getType()) {
            case RECORD:
                reusable.replace(iterationRecord.getValue(), iterationRecord.getTimestamp());
                output.collect(reusable);
                break;
            case EPOCH_WATERMARK:
                LOG.debug(
                        "Head Received epoch watermark {}", iterationRecord.getValue().getEpoch());
                sendEpochWatermarkToCoordinator(iterationRecord.getValue().getEpoch());
                break;
        }
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public void handleOperatorEvent(OperatorEvent operatorEvent) {
        if (operatorEvent instanceof GloballyAlignedEvent) {
            try {
                GloballyAlignedEvent globallyAlignedEvent = (GloballyAlignedEvent) operatorEvent;
                LOG.info("Received global event {}", globallyAlignedEvent);

                shouldTerminate = globallyAlignedEvent.isTerminated();
                reusable.replace(
                        IterationRecord.newEpochWatermark(
                                globallyAlignedEvent.isTerminated()
                                        ? Integer.MAX_VALUE
                                        : globallyAlignedEvent.getEpoch(),
                                uniqueSenderId),
                        0);
                eventBroadcastOutput.broadcastEmit((StreamRecord) reusable);
                numFeedbackRecordsPerEpoch.remove(globallyAlignedEvent.getEpoch());
            } catch (Exception e) {
                ExceptionUtils.rethrow(e);
            }
        }
    }

    @Override
    public void endInput() throws Exception {
        sendEpochWatermarkToCoordinator(0);
        while (!shouldTerminate) {
            mailboxExecutor.yield();
        }
    }

    private void sendEpochWatermarkToCoordinator(int round) {
        operatorEventGateway.sendEventToCoordinator(
                new SubtaskAlignedEvent(
                        round,
                        numFeedbackRecordsPerEpoch.getOrDefault(round, 0L),
                        isCriteriaStream));
    }

    private void registerFeedbackConsumer(Executor mailboxExecutor) {
        int indexOfThisSubtask = getRuntimeContext().getIndexOfThisSubtask();
        int attemptNum = getRuntimeContext().getAttemptNumber();
        FeedbackKey<StreamRecord<IterationRecord<?>>> feedbackKey =
                OperatorUtils.createFeedbackKey(iterationId, feedbackIndex);
        SubtaskFeedbackKey<StreamRecord<IterationRecord<?>>> key =
                feedbackKey.withSubTaskIndex(indexOfThisSubtask, attemptNum);
        FeedbackChannelBroker broker = FeedbackChannelBroker.get();
        FeedbackChannel<StreamRecord<IterationRecord<?>>> channel = broker.getChannel(key);
        OperatorUtils.registerFeedbackConsumer(channel, this, mailboxExecutor);
    }

    @VisibleForTesting
    Map<Integer, Long> getNumFeedbackRecordsPerEpoch() {
        return numFeedbackRecordsPerEpoch;
    }

    @VisibleForTesting
    public OperatorEventGateway getOperatorEventGateway() {
        return operatorEventGateway;
    }

    @VisibleForTesting
    MailboxExecutor getMailboxExecutor() {
        return mailboxExecutor;
    }
}
