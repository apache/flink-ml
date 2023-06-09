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
import org.apache.flink.iteration.checkpoint.Checkpoints;
import org.apache.flink.iteration.checkpoint.CheckpointsBroker;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannel;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannelBroker;
import org.apache.flink.statefun.flink.core.feedback.FeedbackKey;
import org.apache.flink.statefun.flink.core.feedback.SubtaskFeedbackKey;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
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
    private transient Consumer<StreamRecord<IterationRecord<?>>> recordConsumer;

    private transient FeedbackChannel<StreamRecord<IterationRecord<?>>> channel;

    public TailOperator(IterationID iterationId, int feedbackIndex) {
        this.iterationId = Objects.requireNonNull(iterationId);
        this.feedbackIndex = feedbackIndex;
    }

    @Override
    public void setup(
            StreamTask<?, ?> containingTask,
            StreamConfig config,
            Output<StreamRecord<Void>> output) {
        super.setup(containingTask, config, output);
    }

    @Override
    public void open() throws Exception {
        super.open();

        int indexOfThisSubtask = getRuntimeContext().getIndexOfThisSubtask();
        int attemptNum = getRuntimeContext().getAttemptNumber();

        FeedbackKey<StreamRecord<IterationRecord<?>>> feedbackKey =
                OperatorUtils.createFeedbackKey(iterationId, feedbackIndex);
        SubtaskFeedbackKey<StreamRecord<IterationRecord<?>>> key =
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
        recordConsumer.accept(streamRecord);
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        super.prepareSnapshotPreBarrier(checkpointId);
        channel.put(new StreamRecord<>(IterationRecord.newBarrier(checkpointId)));
    }

    @Override
    public void notifyCheckpointAborted(long checkpointId) throws Exception {
        super.notifyCheckpointAborted(checkpointId);

        // TODO: Unfortunately, we have to rely on the tail operator to help
        // abort the checkpoint since the task thread of the head operator
        // might get blocked due to not be able to close the raw state files.
        // We would try to fix it in the Flink side in the future.
        SubtaskFeedbackKey<?> key =
                OperatorUtils.createFeedbackKey(iterationId, feedbackIndex)
                        .withSubTaskIndex(
                                getRuntimeContext().getIndexOfThisSubtask(),
                                getRuntimeContext().getAttemptNumber());
        Checkpoints<?> checkpoints = CheckpointsBroker.get().getCheckpoints(key);
        if (checkpoints != null) {
            checkpoints.abort(checkpointId);
        }
    }

    private void processIfObjectReuseEnabled(StreamRecord<IterationRecord<?>> record) {
        // Since the record would be reused, we have to clone a new one
        IterationRecord<?> cloned = record.getValue().clone();
        cloned.incrementEpoch();
        channel.put(new StreamRecord<>(cloned, record.getTimestamp()));
    }

    private void processIfObjectReuseNotEnabled(StreamRecord<IterationRecord<?>> record) {
        // Since the record would not be reused, we could modify it in place.
        record.getValue().incrementEpoch();
        channel.put(new StreamRecord<>(record.getValue(), record.getTimestamp()));
    }

    @Override
    public void close() throws Exception {
        IOUtils.closeQuietly(channel);
        super.close();
    }
}
