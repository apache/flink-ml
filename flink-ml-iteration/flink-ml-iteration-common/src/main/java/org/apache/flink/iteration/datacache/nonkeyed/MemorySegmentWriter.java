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

package org.apache.flink.iteration.datacache.nonkeyed;

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.core.memory.MemorySegment;
import org.apache.flink.runtime.memory.MemoryAllocationException;
import org.apache.flink.table.runtime.util.MemorySegmentPool;

import javax.annotation.Nullable;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/** A class that writes cache data to memory segments. */
@Internal
class MemorySegmentWriter<T> implements SegmentWriter<T> {

    /** The tool to serialize received records into bytes. */
    private final TypeSerializer<T> serializer;

    /** The pre-allocated path to hold cached records into the file system. */
    private final Path path;

    /** The pool to allocate memory segments from. */
    private final MemorySegmentPool segmentPool;

    /** The output stream to write serialized content to memory segments. */
    private final ManagedMemoryOutputStream outputStream;

    /** The wrapper view of the output stream to be used with TypeSerializer API. */
    private final DataOutputView outputView;

    /** The number of records added so far. */
    private int count;

    MemorySegmentWriter(
            TypeSerializer<T> serializer,
            Path path,
            MemorySegmentPool segmentPool,
            long expectedSize)
            throws MemoryAllocationException {
        this.serializer = serializer;
        this.path = path;
        this.segmentPool = segmentPool;
        this.outputStream = new ManagedMemoryOutputStream(segmentPool, expectedSize);
        this.outputView = new DataOutputViewStreamWrapper(outputStream);
        this.count = 0;
    }

    @Override
    public boolean addRecord(T record) throws IOException {
        if (outputStream.getPos() >= DataCacheWriter.MAX_SEGMENT_SIZE) {
            return false;
        }
        try {
            serializer.serialize(record, outputView);
            count++;
            return true;
        } catch (RuntimeException e) {
            if (e.getCause() instanceof MemoryAllocationException) {
                return false;
            }
            throw e;
        }
    }

    @Override
    public Optional<Segment> finish() throws IOException {
        if (count > 0) {
            return Optional.of(new Segment(path, count, outputStream.getSegments()));
        } else {
            segmentPool.returnAll(outputStream.getSegments());
            return Optional.empty();
        }
    }

    /** An output stream subclass that accepts bytes and writes them to memory segments. */
    private static class ManagedMemoryOutputStream extends OutputStream {

        /** The pool to allocate memory segments from. */
        private final MemorySegmentPool segmentPool;

        /** The number of bytes in a memory segment. */
        private final int pageSize;

        /** The memory segments containing written bytes. */
        private final List<MemorySegment> segments = new ArrayList<>();

        /** The index of the segment that currently accepts written bytes. */
        private int segmentIndex;

        /** The number of bytes in the current segment that have been written. */
        private int segmentOffset;

        /** The number of bytes that have been written so far. */
        private long globalOffset;

        /** The number of bytes that have been allocated so far. */
        private long allocatedBytes;

        public ManagedMemoryOutputStream(MemorySegmentPool segmentPool, long expectedSize)
                throws MemoryAllocationException {
            this.segmentPool = segmentPool;
            this.pageSize = segmentPool.pageSize();
            ensureCapacity(Math.max(expectedSize, 1L));
        }

        public long getPos() {
            return globalOffset;
        }

        public List<MemorySegment> getSegments() {
            return segments;
        }

        @Override
        public void write(int b) throws IOException {
            write(new byte[] {(byte) b}, 0, 1);
        }

        @Override
        public void write(@Nullable byte[] b, int off, int len) throws IOException {
            try {
                ensureCapacity(globalOffset + len);
            } catch (MemoryAllocationException e) {
                throw new RuntimeException(e);
            }

            while (len > 0) {
                int currentLen = Math.min(len, pageSize - segmentOffset);
                segments.get(segmentIndex).put(segmentOffset, b, off, currentLen);
                segmentOffset += currentLen;
                globalOffset += currentLen;
                if (segmentOffset >= pageSize) {
                    segmentIndex++;
                    segmentOffset = 0;
                }
                off += currentLen;
                len -= currentLen;
            }
        }

        private void ensureCapacity(long capacity) throws MemoryAllocationException {
            if (allocatedBytes >= capacity) {
                return;
            }

            int required =
                    (int) (capacity % pageSize == 0 ? capacity / pageSize : capacity / pageSize + 1)
                            - segments.size();

            List<MemorySegment> allocatedSegments = new ArrayList<>();
            for (int i = 0; i < required; i++) {
                MemorySegment memorySegment = segmentPool.nextSegment();
                if (memorySegment == null) {
                    segmentPool.returnAll(allocatedSegments);
                    throw new MemoryAllocationException();
                }
                allocatedSegments.add(memorySegment);
            }

            segments.addAll(allocatedSegments);
            allocatedBytes += (long) allocatedSegments.size() * pageSize;
        }
    }
}
