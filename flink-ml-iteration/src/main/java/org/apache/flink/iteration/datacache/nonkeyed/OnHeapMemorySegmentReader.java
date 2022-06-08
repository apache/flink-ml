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

import java.io.IOException;
import java.util.List;

/** A class that reads data cached in heap memory. */
public class OnHeapMemorySegmentReader<T> implements SegmentReader<T> {

    /** The cached data. */
    private final List<T> list;

    /** The number of records that have been read so far. */
    private int count;

    public OnHeapMemorySegmentReader(Segment segment, int startOffset) {
        this.list = segment.getOnHeapCache();
        this.count = startOffset;
    }

    @Override
    public boolean hasNext() {
        return count < list.size();
    }

    @Override
    public T next() throws IOException {
        return list.get(count++);
    }
}
