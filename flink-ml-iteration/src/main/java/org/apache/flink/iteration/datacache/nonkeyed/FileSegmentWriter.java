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
import org.apache.flink.core.fs.FSDataOutputStream;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.util.Optional;

/** A class that writes cache data to a target file in given file system. */
@Internal
class FileSegmentWriter<T> implements SegmentWriter<T> {

    /** The tool to serialize received records into bytes. */
    private final TypeSerializer<T> serializer;

    /** The path to the target file. */
    private final Path path;

    /** The output stream that writes to the target file. */
    private final FSDataOutputStream outputStream;

    /** A buffer that wraps the output stream to optimize performance. */
    private final BufferedOutputStream bufferedOutputStream;

    /** The wrapper view of the output stream to be used with TypeSerializer API. */
    private final DataOutputView outputView;

    /** The number of records added so far. */
    private int count;

    FileSegmentWriter(TypeSerializer<T> serializer, Path path) throws IOException {
        this.serializer = serializer;
        this.path = path;
        this.outputStream = path.getFileSystem().create(path, FileSystem.WriteMode.NO_OVERWRITE);
        this.bufferedOutputStream = new BufferedOutputStream(outputStream);
        this.outputView = new DataOutputViewStreamWrapper(bufferedOutputStream);
    }

    @Override
    public boolean addRecord(T record) throws IOException {
        if (outputStream.getPos() >= DataCacheWriter.MAX_SEGMENT_SIZE) {
            return false;
        }
        serializer.serialize(record, outputView);
        count++;
        return true;
    }

    @Override
    public Optional<Segment> finish() throws IOException {
        bufferedOutputStream.flush();
        long size = outputStream.getPos();
        outputStream.close();

        if (count > 0) {
            Segment segment = new Segment(path, count, size);
            return Optional.of(segment);
        } else {
            // If there are no records, we tend to directly delete this file
            path.getFileSystem().delete(path, false);
            return Optional.empty();
        }
    }
}
