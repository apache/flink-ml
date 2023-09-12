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
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputSerializer;
import org.apache.flink.core.memory.MemorySegment;
import org.apache.flink.runtime.io.disk.InputViewIterator;
import org.apache.flink.runtime.io.disk.iomanager.IOManager;
import org.apache.flink.runtime.iterative.io.SerializedUpdateBuffer;
import org.apache.flink.runtime.memory.MemoryAllocationException;
import org.apache.flink.runtime.memory.MemoryManager;
import org.apache.flink.util.MutableObjectIterator;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/**
 * * A queue that can spill the items to disks automatically when the memory buffer is full.
 *
 * @param <T> The element type.
 */
@Internal
final class SpillableFeedbackQueue<T> {
    private final DataOutputSerializer output = new DataOutputSerializer(256);
    private final TypeSerializer<T> serializer;
    private final MemoryManager memoryManager;
    private final List<MemorySegment> freeMemory;
    private final SerializedUpdateBuffer buffer;
    private long size = 0L;

    SpillableFeedbackQueue(
            IOManager ioManager,
            MemoryManager memoryManager,
            TypeSerializer<T> serializer,
            long inMemoryBufferSize)
            throws MemoryAllocationException {
        this.serializer = Objects.requireNonNull(serializer);
        this.memoryManager = Objects.requireNonNull(memoryManager);

        int numPages = (int) (inMemoryBufferSize / memoryManager.getPageSize());
        this.freeMemory = memoryManager.allocatePages(this, numPages);
        this.buffer =
                new SerializedUpdateBuffer(freeMemory, memoryManager.getPageSize(), ioManager);
    }

    void add(T item) {
        try {
            output.clear();
            serializer.serialize(item, output);
            buffer.write(output.getSharedBuffer(), 0, output.length());
            size++;
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    MutableObjectIterator<T> iterate() {
        try {
            DataInputView input = buffer.switchBuffers();
            return new InputViewIterator<>(input, this.serializer);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    long size() {
        return size;
    }

    public void reset() {
        this.size = 0;
    }

    void release() {
        output.clear();
        List<MemorySegment> toRelease = buffer.close();
        toRelease.addAll(freeMemory);
        freeMemory.clear();
        memoryManager.release(toRelease);
    }
}
