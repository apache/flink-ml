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

package org.apache.flink.ml.iteration.operator;

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.TaskInfo;
import org.apache.flink.api.common.operators.MailboxExecutor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.ml.iteration.IterationID;
import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.broadcast.BroadcastOutput;
import org.apache.flink.ml.iteration.broadcast.BroadcastOutputFactory;
import org.apache.flink.ml.iteration.operator.event.GloballyAlignedEvent;
import org.apache.flink.ml.iteration.operator.event.SubtaskAlignedEvent;
import org.apache.flink.ml.iteration.operator.headprocessor.HeadOperatorRecordProcessor;
import org.apache.flink.ml.iteration.operator.headprocessor.RegularHeadOperatorRecordProcessor;
import org.apache.flink.ml.iteration.operator.headprocessor.TerminatingHeadOperatorRecordProcessor;
import org.apache.flink.ml.iteration.typeinfo.IterationRecordTypeInfo;
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
import org.apache.flink.util.FlinkRuntimeException;
import org.apache.flink.util.OutputTag;

import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.Executor;

import static org.apache.flink.util.Preconditions.checkState;

/**
 * The head operators unions the initialized variable stream and the feedback stream, and
 * synchronize the epoch watermark (round).
 */
public class HeadOperator extends AbstractStreamOperator<IterationRecord<?>>
        implements OneInputStreamOperator<IterationRecord<?>, IterationRecord<?>>,
                FeedbackConsumer<StreamRecord<IterationRecord<?>>>,
                OperatorEventHandler,
                BoundedOneInput {

    public static final OutputTag<IterationRecord<Void>> ALIGN_NOTIFY_OUTPUT_TAG =
            new OutputTag<>("aligned", new IterationRecordTypeInfo<>(BasicTypeInfo.VOID_TYPE_INFO));

    private final IterationID iterationId;

    private final int feedbackIndex;

    private final boolean isCriteriaStream;

    private final OperatorEventGateway operatorEventGateway;

    private final MailboxExecutor mailboxExecutor;

    private transient BroadcastOutput<?> eventBroadcastOutput;

    private transient ContextImpl processorContext;

    // ------------- runtime -------------------

    private HeadOperatorStatus status;

    private HeadOperatorRecordProcessor recordProcessor;

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
        eventBroadcastOutput =
                BroadcastOutputFactory.createBroadcastOutput(
                        output, metrics.getIOMetricGroup().getNumRecordsOutCounter());
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);

        processorContext = new ContextImpl();
        status = HeadOperatorStatus.RUNNING;
        recordProcessor = new RegularHeadOperatorRecordProcessor(processorContext);

        // Here we register a record
        registerFeedbackConsumer(
                (Runnable runnable) -> {
                    if (status != HeadOperatorStatus.TERMINATED) {
                        mailboxExecutor.execute(runnable::run, "Head feedback");
                    }
                });
    }

    @Override
    public void processElement(StreamRecord<IterationRecord<?>> element) throws Exception {
        recordProcessor.processElement(element);
    }

    @Override
    public void processFeedback(StreamRecord<IterationRecord<?>> iterationRecord) throws Exception {
        boolean terminated = recordProcessor.processFeedbackElement(iterationRecord);
        if (terminated) {
            checkState(status == HeadOperatorStatus.TERMINATING);
            status = HeadOperatorStatus.TERMINATED;
        }
    }

    @Override
    public void handleOperatorEvent(OperatorEvent operatorEvent) {
        if (operatorEvent instanceof GloballyAlignedEvent) {
            boolean shouldTerminate =
                    recordProcessor.onGloballyAligned((GloballyAlignedEvent) operatorEvent);
            if (shouldTerminate) {
                status = HeadOperatorStatus.TERMINATING;
                recordProcessor = new TerminatingHeadOperatorRecordProcessor();
            }
        }
    }

    @Override
    public void endInput() throws Exception {
        recordProcessor.processElement(
                new StreamRecord<>(IterationRecord.newEpochWatermark(0, "fake")));
        while (status != HeadOperatorStatus.TERMINATED) {
            mailboxExecutor.yield();
        }
    }

    private void registerFeedbackConsumer(Executor mailboxExecutor) {
        int indexOfThisSubtask = getRuntimeContext().getIndexOfThisSubtask();
        int attemptNum = getRuntimeContext().getAttemptNumber();
        FeedbackKey<StreamRecord<IterationRecord<?>>> feedbackKey =
                new FeedbackKey<>(iterationId.toHexString(), feedbackIndex);
        SubtaskFeedbackKey<StreamRecord<IterationRecord<?>>> key =
                feedbackKey.withSubTaskIndex(indexOfThisSubtask, attemptNum);
        FeedbackChannelBroker broker = FeedbackChannelBroker.get();
        FeedbackChannel<StreamRecord<IterationRecord<?>>> channel = broker.getChannel(key);
        OperatorUtils.registerFeedbackConsumer(channel, this, mailboxExecutor);
    }

    @VisibleForTesting
    public OperatorEventGateway getOperatorEventGateway() {
        return operatorEventGateway;
    }

    @VisibleForTesting
    MailboxExecutor getMailboxExecutor() {
        return mailboxExecutor;
    }

    @VisibleForTesting
    HeadOperatorRecordProcessor getRecordProcessor() {
        return recordProcessor;
    }

    @VisibleForTesting
    public HeadOperatorStatus getStatus() {
        return status;
    }

    @VisibleForTesting
    enum HeadOperatorStatus {
        RUNNING,

        TERMINATING,

        TERMINATED
    }

    private class ContextImpl implements HeadOperatorRecordProcessor.Context {

        @Override
        public StreamConfig getStreamConfig() {
            return HeadOperator.this.config;
        }

        @Override
        public TaskInfo getTaskInfo() {
            return getContainingTask().getEnvironment().getTaskInfo();
        }

        @Override
        public void output(StreamRecord<IterationRecord<?>> record) {
            output.collect(record);
        }

        @Override
        public void output(
                OutputTag<IterationRecord<?>> outputTag, StreamRecord<IterationRecord<?>> record) {
            output.collect(outputTag, record);
        }

        @Override
        public void broadcastOutput(StreamRecord<IterationRecord<?>> record) {
            try {
                eventBroadcastOutput.broadcastEmit((StreamRecord) record);
            } catch (IOException e) {
                throw new FlinkRuntimeException("Failed to broadcast event", e);
            }
        }

        @Override
        public void updateEpochToCoordinator(int epoch, long numFeedbackRecords) {
            operatorEventGateway.sendEventToCoordinator(
                    new SubtaskAlignedEvent(epoch, numFeedbackRecords, isCriteriaStream));
        }
    }
}
