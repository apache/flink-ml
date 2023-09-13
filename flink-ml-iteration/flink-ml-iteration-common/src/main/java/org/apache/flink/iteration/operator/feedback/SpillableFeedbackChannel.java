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

package org.apache.flink.iteration.operator.feedback;

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.runtime.io.disk.iomanager.IOManager;
import org.apache.flink.runtime.memory.MemoryAllocationException;
import org.apache.flink.runtime.memory.MemoryManager;
import org.apache.flink.statefun.flink.core.feedback.FeedbackConsumer;
import org.apache.flink.statefun.flink.core.feedback.SubtaskFeedbackKey;
import org.apache.flink.util.IOUtils;
import org.apache.flink.util.MutableObjectIterator;
import org.apache.flink.util.Preconditions;

import java.io.Closeable;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Single producer, single consumer channel, which can spill the records to disk when the in-memory
 * buffer is full.
 */
@Internal
public final class SpillableFeedbackChannel<T> implements Closeable {

    /** The key that used to identify this channel. */
    private final SubtaskFeedbackKey<T> key;

    /** A single registered consumer. */
    private final AtomicReference<ConsumerTask<T>> consumerRef = new AtomicReference<>();

    /** The underlying queue used to hold the feedback results. */
    private MpscQueue<T> queue;

    SpillableFeedbackChannel(SubtaskFeedbackKey<T> key) {
        this.key = Objects.requireNonNull(key);
    }

    public void initialize(
            IOManager ioManager,
            MemoryManager memoryManager,
            TypeSerializer<T> serializer,
            long inMemoryBufferSize)
            throws MemoryAllocationException {
        this.queue = new MpscQueue<>(ioManager, memoryManager, serializer, inMemoryBufferSize);
    }

    /** Adds a feedback result to this channel. */
    public void put(T element) {
        if (!isInitialized()) {
            Preconditions.checkState(
                    queue != null,
                    "The SpillableFeedbackChannel has not been initialized, "
                            + "please call SpillableFeedbackChannel#initialize first");
        }
        if (queue.add(element) == 1) {
            final ConsumerTask<T> consumer = consumerRef.get();
            if (consumer != null) {
                consumer.scheduleDrainAll();
            }
        }
    }

    /**
     * Register a feedback iteration consumer.
     *
     * @param consumer the feedback events consumer.
     * @param executor the executor to schedule feedback consumption on.
     */
    public void registerConsumer(final FeedbackConsumer<T> consumer, Executor executor) {
        ConsumerTask<T> consumerTask = new ConsumerTask<>(executor, consumer, queue);
        if (!this.consumerRef.compareAndSet(null, consumerTask)) {
            throw new IllegalStateException(
                    "There can be only a single consumer in a FeedbackChannel.");
        }
        consumerTask.scheduleDrainAll();
    }

    @Override
    public void close() {
        consumerRef.getAndSet(null);
        SpillableFeedbackChannelBroker broker = SpillableFeedbackChannelBroker.get();
        broker.removeChannel(key);
        IOUtils.closeQuietly(queue);
    }

    public boolean isInitialized() {
        return this.queue != null;
    }

    private static final class ConsumerTask<T> implements Runnable {
        private final Executor executor;
        private final FeedbackConsumer<T> consumer;
        private final MpscQueue<T> queue;

        ConsumerTask(Executor executor, FeedbackConsumer<T> consumer, MpscQueue<T> queue) {
            this.executor = Objects.requireNonNull(executor);
            this.consumer = Objects.requireNonNull(consumer);
            this.queue = Objects.requireNonNull(queue);
        }

        void scheduleDrainAll() {
            executor.execute(this);
        }

        @Override
        public void run() {
            final MutableObjectIterator<T> buffer = queue.drainAll();
            try {
                T element;
                while ((element = buffer.next()) != null) {
                    consumer.processFeedback(element);
                }
                queue.resetStandBy();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
