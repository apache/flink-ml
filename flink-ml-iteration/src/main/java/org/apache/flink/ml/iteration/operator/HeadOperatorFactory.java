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
import org.apache.flink.ml.iteration.IterationID;
import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.operator.coordinator.HeadOperatorCoordinator;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.operators.coordination.OperatorCoordinator;
import org.apache.flink.runtime.operators.coordination.OperatorEventGateway;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.CoordinatedOperatorFactory;
import org.apache.flink.streaming.api.operators.MailboxExecutor;
import org.apache.flink.streaming.api.operators.OneInputStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.runtime.tasks.mailbox.TaskMailbox;

/** The Factory for the {@link HeadOperator}. */
public class HeadOperatorFactory extends AbstractStreamOperatorFactory<IterationRecord<?>>
        implements OneInputStreamOperatorFactory<IterationRecord<?>, IterationRecord<?>>,
                CoordinatedOperatorFactory<IterationRecord<?>> {

    private final IterationID iterationId;

    private final int feedbackIndex;

    private final boolean isCriteriaStream;

    private final int totalHeadParallelism;

    public HeadOperatorFactory(
            IterationID iterationId,
            int feedbackIndex,
            boolean isCriteriaStream,
            int totalHeadParallelism) {
        this.iterationId = iterationId;
        this.feedbackIndex = feedbackIndex;
        this.isCriteriaStream = isCriteriaStream;
        this.totalHeadParallelism = totalHeadParallelism;
    }

    @Override
    public <T extends StreamOperator<IterationRecord<?>>> T createStreamOperator(
            StreamOperatorParameters<IterationRecord<?>> streamOperatorParameters) {

        OperatorID operatorId = streamOperatorParameters.getStreamConfig().getOperatorID();

        // TODO: We would have to create a new mailboxExecutor since the given one
        // is created with getChainedIndex as the priority, which seems to be a bug.
        MailboxExecutor mailboxExecutor =
                streamOperatorParameters
                        .getContainingTask()
                        .getMailboxExecutorFactory()
                        .createExecutor(TaskMailbox.MIN_PRIORITY);

        HeadOperator headOperator =
                new HeadOperator(
                        iterationId,
                        feedbackIndex,
                        isCriteriaStream,
                        mailboxExecutor,
                        createOperatorEventGateway(streamOperatorParameters),
                        streamOperatorParameters.getProcessingTimeService());
        headOperator.setup(
                streamOperatorParameters.getContainingTask(),
                streamOperatorParameters.getStreamConfig(),
                streamOperatorParameters.getOutput());
        streamOperatorParameters
                .getOperatorEventDispatcher()
                .registerEventHandler(
                        streamOperatorParameters.getStreamConfig().getOperatorID(), headOperator);
        return (T) headOperator;
    }

    @Override
    public OperatorCoordinator.Provider getCoordinatorProvider(String s, OperatorID operatorID) {
        return new HeadOperatorCoordinator.HeadOperatorCoordinatorProvider(
                operatorID, iterationId, totalHeadParallelism);
    }

    @Override
    public Class<? extends StreamOperator> getStreamOperatorClass(ClassLoader classLoader) {
        return HeadOperator.class;
    }

    @VisibleForTesting
    OperatorEventGateway createOperatorEventGateway(
            StreamOperatorParameters<IterationRecord<?>> streamOperatorParameters) {
        return streamOperatorParameters
                .getOperatorEventDispatcher()
                .getOperatorEventGateway(
                        streamOperatorParameters.getStreamConfig().getOperatorID());
    }
}
