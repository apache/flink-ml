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

import org.apache.flink.core.fs.Path;

import org.openjdk.jol.info.GraphLayout;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/** A class that writes cache data to heap memory. */
public class OnHeapMemorySegmentWriter<T> implements SegmentWriter<T> {

    /** The pre-allocated path to hold cached records into the file system. */
    private final Path path;

    /** The pool to allocate heap memory space from. */
    private final OnHeapMemoryPool memoryPool;

    /** The cached data. */
    private final List<T> list = new ArrayList<>();

    public OnHeapMemorySegmentWriter(Path path, OnHeapMemoryPool memoryPool) {
        this.path = path;
        this.memoryPool = memoryPool;
    }

    @Override
    public boolean addRecord(T record) throws IOException {
        long recordSize = GraphLayout.parseInstance(record).totalSize();
        if (!memoryPool.acquireMemory(recordSize)) {
            return false;
        }
        list.add(record);
        return true;
    }

    @Override
    public Optional<Segment> finish() throws IOException {
        if (list.size() > 0) {
            return Optional.of(new Segment(path, list));
        }
        return Optional.empty();
    }
}
