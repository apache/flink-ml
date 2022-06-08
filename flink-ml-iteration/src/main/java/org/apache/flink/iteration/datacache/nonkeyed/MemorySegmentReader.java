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
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.MemorySegment;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/** A class that reads data cached in memory. */
@Internal
class MemorySegmentReader<T> implements SegmentReader<T> {

    /** The tool to deserialize bytes into records. */
    private final TypeSerializer<T> serializer;

    /** The wrapper view of the input stream of memory segments to be used in TypeSerializer API. */
    private final DataInputView inputView;

    /** The total number of records contained in the segments. */
    private final int totalCount;

    /** The number of records that have been read so far. */
    private int count;

    MemorySegmentReader(TypeSerializer<T> serializer, Segment segment, int startOffset)
            throws IOException {
        ManagedMemoryInputStream inputStream = new ManagedMemoryInputStream(segment.getCache());
        this.inputView = new DataInputViewStreamWrapper(inputStream);
        this.serializer = serializer;
        this.totalCount = segment.getCount();
        this.count = 0;

        for (int i = 0; i < startOffset; i++) {
            next();
        }
    }

    @Override
    public boolean hasNext() {
        return count < totalCount;
    }

    @Override
    public T next() throws IOException {
        T value = serializer.deserialize(inputView);
        count++;
        return value;
    }

    @Override
    public void close() {}

    /** An input stream subclass that reads bytes from memory segments. */
    private static class ManagedMemoryInputStream extends InputStream {

        /** The memory segments to read bytes from. */
        private final List<MemorySegment> segments;

        /** The index of the segment that is currently being read. */
        private int segmentIndex;

        /** The number of bytes that have been read from the current segment so far. */
        private int segmentOffset;

        public ManagedMemoryInputStream(List<MemorySegment> segments) {
            this.segments = segments;
            this.segmentIndex = 0;
            this.segmentOffset = 0;
        }

        @Override
        public int read() throws IOException {
            int ret = segments.get(segmentIndex).get(segmentOffset) & 0xff;
            segmentOffset += 1;
            if (segmentOffset >= segments.get(segmentIndex).size()) {
                segmentIndex++;
                segmentOffset = 0;
            }
            return ret;
        }

        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int readLen = 0;

            while (len > 0 && segmentIndex < segments.size()) {
                int currentLen = Math.min(segments.get(segmentIndex).size() - segmentOffset, len);
                segments.get(segmentIndex).get(segmentOffset, b, off, currentLen);
                segmentOffset += currentLen;
                if (segmentOffset >= segments.get(segmentIndex).size()) {
                    segmentIndex++;
                    segmentOffset = 0;
                }

                readLen += currentLen;
                off += currentLen;
                len -= currentLen;
            }

            return readLen;
        }
    }
}
