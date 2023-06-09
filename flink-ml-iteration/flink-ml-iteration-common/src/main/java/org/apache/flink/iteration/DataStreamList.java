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

package org.apache.flink.iteration;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** A utility class to maintain a list of {@link DataStream}, which might have different types. */
@Experimental
public class DataStreamList {

    public static DataStreamList of(DataStream<?>... streams) {
        return new DataStreamList(Arrays.asList(streams));
    }

    private final List<DataStream<?>> dataStreams;

    public DataStreamList(List<DataStream<?>> dataStreams) {
        this.dataStreams = Collections.unmodifiableList(dataStreams);
    }

    /** Returns the number of data streams in this list. */
    public int size() {
        return dataStreams.size();
    }

    /** Returns the data stream at the given index in this list. */
    @SuppressWarnings("unchecked")
    public <T> DataStream<T> get(int index) {
        return (DataStream<T>) dataStreams.get(index);
    }

    /** Returns all the data streams as a native list. */
    public List<DataStream<?>> getDataStreams() {
        return dataStreams;
    }
}
