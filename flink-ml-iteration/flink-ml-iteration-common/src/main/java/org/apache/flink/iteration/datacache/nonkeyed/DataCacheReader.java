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

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

/** Reads the cached data from a list of segments. */
public class DataCacheReader<T> implements Iterator<T> {

    /** The tool to deserialize bytes into records. */
    private final TypeSerializer<T> serializer;

    /** The segments where to read the records from. */
    private final List<Segment> segments;

    /** The current reader for next records. */
    @Nullable private SegmentReader<T> currentSegmentReader;

    /** The index of the segment that current reader reads from. */
    private int currentSegmentIndex;

    /** The number of records that have been read through the current reader so far. */
    private int currentSegmentCount;

    public DataCacheReader(TypeSerializer<T> serializer, List<Segment> segments) {
        this(serializer, segments, Tuple2.of(0, 0));
    }

    public DataCacheReader(
            TypeSerializer<T> serializer,
            List<Segment> segments,
            Tuple2<Integer, Integer> readerPosition) {
        this.serializer = serializer;
        this.segments = segments;
        this.currentSegmentIndex = readerPosition.f0;
        this.currentSegmentCount = readerPosition.f1;

        createSegmentReader(readerPosition.f0, readerPosition.f1);
    }

    @Override
    public boolean hasNext() {
        return currentSegmentReader != null && currentSegmentReader.hasNext();
    }

    @Override
    public T next() {
        try {
            T record = currentSegmentReader.next();

            currentSegmentCount++;
            if (!currentSegmentReader.hasNext()) {
                currentSegmentReader.close();
                currentSegmentIndex++;
                currentSegmentCount = 0;
                createSegmentReader(currentSegmentIndex, currentSegmentCount);
            }

            return record;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Tuple2<Integer, Integer> getPosition() {
        return new Tuple2<>(currentSegmentIndex, currentSegmentCount);
    }

    private void createSegmentReader(int index, int startOffset) {
        try {
            if (index >= segments.size()) {
                currentSegmentReader = null;
                return;
            }

            Segment segment = segments.get(currentSegmentIndex);
            if (!segment.getCache().isEmpty()) {
                currentSegmentReader = new MemorySegmentReader<>(serializer, segment, startOffset);
            } else {
                currentSegmentReader = new FileSegmentReader<>(serializer, segment, startOffset);
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
