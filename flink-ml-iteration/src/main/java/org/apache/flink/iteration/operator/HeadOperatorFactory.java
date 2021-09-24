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
import org.apache.flink.iteration.operator.coordinator.HeadOperatorCoordinator;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.operators.coordination.OperatorCoordinator;
import org.apache.flink.runtime.operators.coordination.OperatorEventGateway;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.CoordinatedOperatorFactory;
import org.apache.flink.streaming.api.operators.OneInputStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.YieldingOperatorFactory;
import org.apache.flink.streaming.runtime.tasks.mailbox.TaskMailbox;

import static org.apache.flink.util.Preconditions.checkArgument;

/** The Factory for the {@link HeadOperator}. */
public class HeadOperatorFactory extends AbstractStreamOperatorFactory<IterationRecord<?>>
        implements OneInputStreamOperatorFactory<IterationRecord<?>, IterationRecord<?>>,
                CoordinatedOperatorFactory<IterationRecord<?>>,
                YieldingOperatorFactory<IterationRecord<?>> {

    private final IterationID iterationId;

    private final int feedbackIndex;

    private final boolean isCriteriaStream;

    private final int totalInitialVariableParallelism;

    private int criteriaStreamParallelism;

    public HeadOperatorFactory(
            IterationID iterationId,
            int feedbackIndex,
            boolean isCriteriaStream,
            int totalInitialVariableParallelism) {
        this.iterationId = iterationId;
        this.feedbackIndex = feedbackIndex;
        this.isCriteriaStream = isCriteriaStream;

        checkArgument(
                totalInitialVariableParallelism > 0,
                "totalInitialVariableParallelism should be positive");
        this.totalInitialVariableParallelism = totalInitialVariableParallelism;
    }

    public void setCriteriaStreamParallelism(int criteriaStreamParallelism) {
        checkArgument(
                criteriaStreamParallelism > 0,
                "totalInitialVariableParallelism should be positive");
        this.criteriaStreamParallelism = criteriaStreamParallelism;
    }

    @Override
    public <T extends StreamOperator<IterationRecord<?>>> T createStreamOperator(
            StreamOperatorParameters<IterationRecord<?>> streamOperatorParameters) {

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
                operatorID,
                iterationId,
                totalInitialVariableParallelism + criteriaStreamParallelism);
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

    @Override
    public void setMailboxExecutor(MailboxExecutor mailboxExecutor) {
        // We need it to be yielding operator factory to disable chaining,
        // but we cannot use the given mailbox here since it has bugs.
    }
}
