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
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.MemorySegment;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.apache.flink.util.Preconditions.checkArgument;
import static org.apache.flink.util.Preconditions.checkNotNull;

/** A segment contains the information about a cache unit. */
@Internal
public class Segment {

    /** The path to the file containing persisted records. */
    private final Path path;

    /**
     * The count of records in the file at the path if the file size is not zero, otherwise the
     * count of records in the cache.
     */
    private final int count;

    /**
     * The total length of the file containing persisted records. Its value is 0 iff the segment has
     * not been written to the given path.
     */
    private long fsSize = 0L;

    /**
     * The memory segments containing cached records. This list is empty iff the segment has not
     * been cached in memory.
     */
    private List<MemorySegment> cache = new ArrayList<>();

    Segment(Path path, int count, long fsSize) {
        this.path = path;
        this.count = count;
        this.fsSize = fsSize;

        checkNotNull(path);
        checkArgument(count > 0);
        checkArgument(fsSize > 0);
    }

    Segment(Path path, int count, List<MemorySegment> cache) {
        this.path = path;
        this.count = count;
        this.cache = cache;

        checkNotNull(path);
        checkArgument(count > 0);
    }

    void setCache(List<MemorySegment> cache) {
        this.cache = cache;
    }

    void setFsSize(long fsSize) {
        checkArgument(fsSize > 0);
        this.fsSize = fsSize;
    }

    Path getPath() {
        return path;
    }

    int getCount() {
        return count;
    }

    long getFsSize() {
        return fsSize;
    }

    List<MemorySegment> getCache() {
        return cache;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (!(o instanceof Segment)) {
            return false;
        }

        Segment segment = (Segment) o;
        return count == segment.count && Objects.equals(path, segment.path);
    }

    @Override
    public int hashCode() {
        return Objects.hash(path, count);
    }

    @Override
    public String toString() {
        return "Segment{" + "path=" + path + ", count=" + count + '}';
    }
}
