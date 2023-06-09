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

import org.apache.flink.streaming.api.datastream.DataStream;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** A list of data streams and whether they need replayed. */
public class ReplayableDataStreamList {

    private final List<DataStream<?>> replayedDataStreams;

    private final List<DataStream<?>> nonReplayedStreams;

    private ReplayableDataStreamList(
            List<DataStream<?>> replayedDataStreams, List<DataStream<?>> nonReplayedStreams) {
        this.replayedDataStreams = replayedDataStreams;
        this.nonReplayedStreams = nonReplayedStreams;
    }

    public static ReplayedDataStreamList replay(DataStream<?>... dataStreams) {
        return new ReplayedDataStreamList(Arrays.asList(dataStreams));
    }

    public static NonReplayedDataStreamList notReplay(DataStream<?>... dataStreams) {
        return new NonReplayedDataStreamList(Arrays.asList(dataStreams));
    }

    List<DataStream<?>> getReplayedDataStreams() {
        return Collections.unmodifiableList(replayedDataStreams);
    }

    List<DataStream<?>> getNonReplayedStreams() {
        return Collections.unmodifiableList(nonReplayedStreams);
    }

    /** A special {@link ReplayableDataStreamList} that all streams should be replayed. */
    public static class ReplayedDataStreamList extends ReplayableDataStreamList {

        public ReplayedDataStreamList(List<DataStream<?>> replayedDataStreams) {
            super(replayedDataStreams, Collections.emptyList());
        }

        public ReplayableDataStreamList andNotReplay(DataStream<?>... nonReplayedStreams) {
            return new ReplayableDataStreamList(
                    getReplayedDataStreams(), Arrays.asList(nonReplayedStreams));
        }
    }

    /** A special {@link ReplayableDataStreamList} that all streams should be not replayed. */
    public static class NonReplayedDataStreamList extends ReplayableDataStreamList {

        public NonReplayedDataStreamList(List<DataStream<?>> nonReplayedDataStreams) {
            super(Collections.emptyList(), nonReplayedDataStreams);
        }
    }
}
