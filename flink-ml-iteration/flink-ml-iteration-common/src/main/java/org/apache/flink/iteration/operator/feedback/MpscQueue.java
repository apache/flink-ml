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
import org.apache.flink.runtime.util.EmptyMutableObjectIterator;
import org.apache.flink.statefun.flink.core.queue.Lock;
import org.apache.flink.statefun.flink.core.queue.Locks;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.MutableObjectIterator;
import org.apache.flink.util.Preconditions;

import java.io.Closeable;
import java.io.IOException;

/**
 * Multi producers single consumer fifo queue.
 *
 * @param <T> The element type.
 */
@Internal
final class MpscQueue<T> implements Closeable {
    private final Lock lock = Locks.spinLock();

    private SpillableFeedbackQueue<T> activeQueue;
    private SpillableFeedbackQueue<T> standByQueue;

    MpscQueue(
            IOManager ioManager,
            MemoryManager memoryManager,
            TypeSerializer<T> serializer,
            long inMemoryBufferSize)
            throws MemoryAllocationException {
        this.activeQueue =
                new SpillableFeedbackQueue<>(
                        ioManager, memoryManager, serializer, inMemoryBufferSize / 2);
        this.standByQueue =
                new SpillableFeedbackQueue<>(
                        ioManager, memoryManager, serializer, inMemoryBufferSize / 2);
    }

    /**
     * Adds an element to this (unbound) queue.
     *
     * @param element the element to add.
     * @return the number of elements in the queue after the addition.
     */
    long add(T element) {
        Preconditions.checkState(element instanceof StreamRecord);
        final Lock lock = this.lock;
        lock.lockUninterruptibly();
        try {
            SpillableFeedbackQueue<T> active = this.activeQueue;

            active.add(element);
            return active.size();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Atomically drains the queue.
     *
     * @return a batch of elements that obtained atomically from that queue.
     */
    MutableObjectIterator<T> drainAll() {
        final Lock lock = this.lock;
        lock.lockUninterruptibly();
        try {
            final SpillableFeedbackQueue<T> ready = this.activeQueue;
            if (ready.size() == 0) {
                return EmptyMutableObjectIterator.get();
            }
            this.activeQueue = this.standByQueue;
            this.standByQueue = ready;
            return ready.iterate();
        } finally {
            lock.unlock();
        }
    }

    void resetStandBy() throws Exception {
        final Lock lock = this.lock;
        lock.lockUninterruptibly();
        try {
            standByQueue.reset();
        } finally {
            lock.unlock();
        }
    }

    public void close() throws IOException {
        final Lock lock = this.lock;
        lock.lockUninterruptibly();
        try {
            activeQueue.release();
            standByQueue.release();
        } finally {
            lock.unlock();
        }
    }
}
