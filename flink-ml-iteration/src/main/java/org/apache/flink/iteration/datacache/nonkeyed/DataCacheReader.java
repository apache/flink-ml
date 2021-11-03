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

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

/** Reads the cached data from a list of paths. */
public class DataCacheReader<T> implements Iterator<T> {

    private final TypeSerializer<T> serializer;

    private final FileSystem fileSystem;

    private final List<Segment> segments;

    @Nullable private SegmentReader currentSegmentReader;

    public DataCacheReader(
            TypeSerializer<T> serializer, FileSystem fileSystem, List<Segment> segments)
            throws IOException {
        this(serializer, fileSystem, segments, new Tuple2<>(0, 0));
    }

    public DataCacheReader(
            TypeSerializer<T> serializer,
            FileSystem fileSystem,
            List<Segment> segments,
            Tuple2<Integer, Integer> readerPosition)
            throws IOException {

        this.serializer = serializer;
        this.fileSystem = fileSystem;
        this.segments = segments;

        if (readerPosition.f0 < segments.size()) {
            this.currentSegmentReader = new SegmentReader(readerPosition.f0, readerPosition.f1);
        }
    }

    @Override
    public boolean hasNext() {
        return currentSegmentReader != null && currentSegmentReader.hasNext();
    }

    @Override
    public T next() {
        try {
            T next = currentSegmentReader.next();

            if (!currentSegmentReader.hasNext()) {
                currentSegmentReader.close();
                if (currentSegmentReader.index < segments.size() - 1) {
                    currentSegmentReader = new SegmentReader(currentSegmentReader.index + 1, 0);
                } else {
                    currentSegmentReader = null;
                }
            }

            return next;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Tuple2<Integer, Integer> getPosition() {
        if (currentSegmentReader == null) {
            return new Tuple2<>(segments.size(), 0);
        }

        return new Tuple2<>(currentSegmentReader.getIndex(), currentSegmentReader.getOffset());
    }

    private class SegmentReader {

        private final int index;

        private final FSDataInputStream inputStream;

        private final DataInputView inputView;

        private int offset;

        public SegmentReader(int index, int startOffset) throws IOException {
            this.index = index;
            this.inputStream = fileSystem.open(segments.get(index).getPath());
            this.inputView = new DataInputViewStreamWrapper(inputStream);
            this.offset = startOffset;
        }

        public boolean hasNext() {
            return offset < segments.get(index).getCount();
        }

        public T next() throws IOException {
            T next = serializer.deserialize(inputView);
            offset++;
            return next;
        }

        public void close() throws IOException {
            inputStream.close();
        }

        public int getIndex() {
            return index;
        }

        public int getOffset() {
            return offset;
        }
    }
}
