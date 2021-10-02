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

package org.apache.flink.ml.iteration.datacache.nonkeyed;

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.fs.FSDataOutputStream;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/** Records the data received and replayed them on required. */
public class DataCacheWriter<T> {

    private final TypeSerializer<T> serializer;

    private final FileSystem fileSystem;

    private final Supplier<Path> pathGenerator;

    private final List<Segment> finishSegments;

    private Path currentPath;

    private FSDataOutputStream outputStream;

    private DataOutputView outputView;

    private int currentSegmentCount;

    public DataCacheWriter(
            TypeSerializer<T> serializer, FileSystem fileSystem, Supplier<Path> pathGenerator)
            throws IOException {
        this.serializer = serializer;
        this.fileSystem = fileSystem;
        this.pathGenerator = pathGenerator;

        this.finishSegments = new ArrayList<>();

        startNewSegment();
    }

    public void addRecord(T record) throws IOException {
        serializer.serialize(record, outputView);
        currentSegmentCount += 1;
    }

    public List<Segment> finishAddingRecords() throws IOException {
        finishCurrentSegment();
        return finishSegments;
    }

    public List<Segment> getFinishSegments() {
        return finishSegments;
    }

    @VisibleForTesting
    void startNewSegment() throws IOException {
        this.currentPath = pathGenerator.get();
        this.outputStream = fileSystem.create(currentPath, FileSystem.WriteMode.NO_OVERWRITE);
        this.outputView = new DataOutputViewStreamWrapper(outputStream);
        this.currentSegmentCount = 0;
    }

    @VisibleForTesting
    void finishCurrentSegment() throws IOException {
        this.outputStream.close();
        if (currentSegmentCount > 0) {
            finishSegments.add(new Segment(currentPath, currentSegmentCount));
        }
    }
}
