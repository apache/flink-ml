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
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;

import java.io.BufferedInputStream;
import java.io.IOException;

/** A class that reads the cached data in a segment from file system. */
@Internal
class FileSegmentReader<T> implements SegmentReader<T> {

    /** The tool to deserialize bytes into records. */
    private final TypeSerializer<T> serializer;

    /** The input stream to read serialized records from. */
    private final FSDataInputStream inputStream;

    /** The wrapper view of input stream to be used with TypeSerializer API. */
    private final DataInputView inputView;

    /** The total number of records contained in the segments. */
    private final int totalCount;

    /** The number of records that have been read so far. */
    private int count;

    FileSegmentReader(TypeSerializer<T> serializer, Segment segment, int startOffset)
            throws IOException {
        this.serializer = serializer;
        Path path = segment.getPath();
        this.inputStream = path.getFileSystem().open(path);
        BufferedInputStream bufferedInputStream = new BufferedInputStream(inputStream);
        this.inputView = new DataInputViewStreamWrapper(bufferedInputStream);
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
    public void close() throws IOException {
        inputStream.close();
    }
}
