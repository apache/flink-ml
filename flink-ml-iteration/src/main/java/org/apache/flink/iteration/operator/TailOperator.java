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

import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannel;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannelBroker;
import org.apache.flink.statefun.flink.core.feedback.FeedbackKey;
import org.apache.flink.statefun.flink.core.feedback.SubtaskFeedbackKey;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.IOUtils;

import java.util.Objects;
import java.util.function.Consumer;

/**
 * The tail operators is attached after each feedback operator to increment the round of each
 * record.
 */
public class TailOperator extends AbstractStreamOperator<Void>
        implements OneInputStreamOperator<IterationRecord<?>, Void> {

    private final IterationID iterationId;

    private final int feedbackIndex;

    /** We distinguish how the record is processed according to if objectReuse is enabled. */
    private transient Consumer<IterationRecord<?>> recordConsumer;

    private transient FeedbackChannel<IterationRecord<?>> channel;

    public TailOperator(IterationID iterationId, int feedbackIndex) {
        this.iterationId = Objects.requireNonNull(iterationId);
        this.feedbackIndex = feedbackIndex;
    }

    @Override
    public void open() throws Exception {
        super.open();

        int indexOfThisSubtask = getRuntimeContext().getIndexOfThisSubtask();
        int attemptNum = getRuntimeContext().getAttemptNumber();

        FeedbackKey<IterationRecord<?>> feedbackKey =
                OperatorUtils.createFeedbackKey(iterationId, feedbackIndex);
        SubtaskFeedbackKey<IterationRecord<?>> key =
                feedbackKey.withSubTaskIndex(indexOfThisSubtask, attemptNum);

        FeedbackChannelBroker broker = FeedbackChannelBroker.get();
        this.channel = broker.getChannel(key);

        this.recordConsumer =
                getExecutionConfig().isObjectReuseEnabled()
                        ? this::processIfObjectReuseEnabled
                        : this::processIfObjectReuseNotEnabled;
    }

    @Override
    public void processElement(StreamRecord<IterationRecord<?>> streamRecord) {
        recordConsumer.accept(streamRecord.getValue());
    }

    private void processIfObjectReuseEnabled(IterationRecord<?> record) {
        // Since the record would be reused, we have to clone a new one
        IterationRecord<?> cloned = record.clone();
        cloned.incrementEpoch();
        channel.put(cloned);
    }

    private void processIfObjectReuseNotEnabled(IterationRecord<?> record) {
        // Since the record would not be reused, we could modify it in place.
        record.incrementEpoch();
        channel.put(record);
    }

    @Override
    public void close() throws Exception {
        IOUtils.closeQuietly(channel);
        super.close();
    }
}
