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

import javax.annotation.Nullable;

/** The result of an iteration, specifying the feedbacks and the outputs. */
@Experimental
public class IterationBodyResult {

    /**
     * A list of feedback variable streams. These streams will only be used during the iteration
     * execution and will not be returned to the caller of the iteration body. It is assumed that
     * the method which executes the iteration body will feed the records of the feedback variable
     * streams back to the corresponding input variable streams.
     */
    private final DataStreamList feedbackVariableStreams;

    /**
     * A list of output streams. These streams will be returned to the caller of the methods that
     * execute the iteration body.
     */
    private final DataStreamList outputStreams;

    /**
     * An optional termination criteria stream. If this stream is not null, it will be used together
     * with the feedback variable streams to determine when the iteration should terminate.
     */
    private final @Nullable DataStream<?> terminationCriteria;

    public IterationBodyResult(
            DataStreamList feedbackVariableStreams, DataStreamList outputStreams) {
        this(feedbackVariableStreams, outputStreams, null);
    }

    public IterationBodyResult(
            DataStreamList feedbackVariableStreams,
            DataStreamList outputStreams,
            @Nullable DataStream<?> terminationCriteria) {
        this.feedbackVariableStreams = feedbackVariableStreams;
        this.outputStreams = outputStreams;
        this.terminationCriteria = terminationCriteria;
    }

    public DataStreamList getFeedbackVariableStreams() {
        return feedbackVariableStreams;
    }

    public DataStreamList getOutputStreams() {
        return outputStreams;
    }

    @Nullable
    public DataStream<?> getTerminationCriteria() {
        return terminationCriteria;
    }
}
