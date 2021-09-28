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

package org.apache.flink.ml.iteration;

import org.apache.flink.annotation.PublicEvolving;

/** A helper class to create iterations. */
@PublicEvolving
public class Iterations {

    /**
     * This method uses an iteration body to process records in unbounded data streams.
     *
     * <p>This method invokes the iteration body with the following parameters: 1) The 1st parameter
     * is a list of input variable streams, which are created as the union of the initial variable
     * streams and the corresponding feedback variable streams (returned by the iteration body). 2)
     * The 2nd parameter is the data streams given to this method.
     *
     * <p>The epoch values are determined as described below. See IterationListener for how the
     * epoch values are used. 1) All records in the initial variable streams and initial data
     * streams has epoch=0. 2) For any record emitted by this operator into a non-feedback stream,
     * the epoch of this emitted record = the epoch of the input record that triggers this emission.
     * If this record is emitted by onEpochWatermarkIncremented(), then the epoch of this record =
     * epochWatermark. 3) For any record emitted by this operator into a feedback variable stream,
     * the epoch of the emitted record = the epoch of the input record that triggers this emission +
     * 1. If this record is emitted by onEpochWatermarkIncremented(), then the epoch of this record
     * = epochWatermark.
     *
     * <p>The iteration would not terminate if at least one of its inputs is unbounded. Otherwise it
     * will terminated after all the inputs are terminated and no more records are iterating.
     *
     * <p>Required: 1) The parallelism of any stream in the initial variable streams must equal to
     * the parallelism of the stream at the same index of the feedback variable streams returned by
     * the IterationBody.
     *
     * @param initVariableStreams The initial variable streams. These streams will be merged with
     *     the feedback variable streams before being used as the 1st parameter to invoke the
     *     iteration body.
     * @param dataStreams The data streams. These streams will be used as the 2nd parameter to
     *     invoke the iteration body.
     * @param body The computation logic which takes variable/data streams and returns
     *     variable/output streams.
     * @return The list of output streams returned by the iteration boy.
     */
    public static DataStreamList iterateUnboundedStreams(
            DataStreamList initVariableStreams, DataStreamList dataStreams, IterationBody body) {
        return null;
    }

    /**
     * This method uses an iteration body to process records in some bounded data streams
     * iteratively until a termination criteria is reached (e.g. the given number of rounds is
     * completed or no further variable update is needed). Because this method does not replay
     * records in the data streams, the iteration body needs to cache those records in order to
     * visit those records repeatedly.
     *
     * <p>This method invokes the iteration body with the following parameters: 1) The 1st parameter
     * is a list of input variable streams, which are created as the union of the initial variable
     * streams and the corresponding feedback variable streams (returned by the iteration body). 2)
     * The 2nd parameter is the data streams given to this method.
     *
     * <p>The epoch values are determined as described below. See IterationListener for how the
     * epoch values are used. 1) All records in the initial variable streams has epoch=0. 2) All
     * records in the data streams has epoch=0. 3) For any record emitted by this operator into a
     * non-feedback stream, the epoch of this emitted record = the epoch of the input record that
     * triggers this emission. If this record is emitted by onEpochWatermarkIncremented(), then the
     * epoch of this record = epochWatermark. 4) For any record emitted by this operator into a
     * feedback variable stream, the epoch of the emitted record = the epoch of the input record
     * that triggers this emission + 1.
     *
     * <p>Suppose there is a coordinator operator which takes all feedback variable streams (emitted
     * by the iteration body) and the termination criteria stream (if not null) as inputs. The
     * execution of the graph created by the iteration body will terminate when all input streams
     * have been fully consumed AND any of the following conditions is met: 1) The termination
     * criteria stream is not null. And the coordinator operator has not observed any new value from
     * the termination criteria stream between two consecutive onEpochWatermarkIncremented
     * invocations. 2) The coordinator operator has not observed any new value from any feedback
     * variable stream between two consecutive onEpochWatermarkIncremented invocations.
     *
     * <p>Required: 1) All the init variable streams and the data streams must be bounded. 2) The
     * parallelism of any stream in the initial variable streams must equal the parallelism of the
     * stream at the same index of the feedback variable streams returned by the IterationBody.
     *
     * @param initVariableStreams The initial variable streams. These streams will be merged with
     *     the feedback variable streams before being used as the 1st parameter to invoke the
     *     iteration body.
     * @param dataStreams The data streams. These streams will be used as the 2nd parameter to
     *     invoke the iteration body.
     * @param body The computation logic which takes variable/data streams and returns
     *     variable/output streams.
     * @return The list of output streams returned by the iteration boy.
     */
    public static DataStreamList iterateBoundedStreamsUntilTermination(
            DataStreamList initVariableStreams, DataStreamList dataStreams, IterationBody body) {
        return null;
    }

    /**
     * This method can use an iteration body to process records in some bounded data streams
     * iteratively until a termination criteria is reached (e.g. the given number of rounds is
     * completed or no further variable update is needed). Because this method replays records in
     * the data streams, the iteration body does not need to cache those records to visit those
     * records repeatedly.
     *
     * <p>This method invokes the iteration body with the following parameters: 1) The 1st parameter
     * is a list of input variable streams, which are created as the union of the initial variable
     * streams and the corresponding feedback variable streams (returned by the iteration body). 2)
     * The 2nd parameter is a list of replayed data streams, which are created by replaying the
     * initial data streams round by round until the iteration terminates. The records in the Nth
     * round will be emitted into the iteration body only if the low watermark of the first operator
     * in the iteration body >= N - 1.
     *
     * <p>The epoch values are determined as described below. See IterationListener for how the
     * epoch values are used. 1) All records in the initial variable streams has epoch=0. 2) The
     * records from the initial data streams will be replayed round by round into the iteration
     * body. The records in the first round have epoch=0. And records in the Nth round have epoch =
     * N. 3) For any record emitted by this operator into a non-feedback stream, the epoch of this
     * emitted record = the epoch of the input record that triggers this emission. If this record is
     * emitted by onEpochWatermarkIncremented(), then the epoch of this record = epochWatermark. 4)
     * For any record emitted by this operator into a feedback stream, the epoch of the emitted
     * record = the epoch of the input record that triggers this emission + 1.
     *
     * <p>Suppose there is a coordinator operator which takes all feedback variable streams (emitted
     * by the iteration body) and the termination criteria stream (if not null) as inputs. The
     * execution of the graph created by the iteration body will terminate when all input streams
     * have been fully consumed AND any of the following conditions is met: 1) The termination
     * criteria stream is not null. And the coordinator operator has not observed any new value from
     * the termination criteria stream between two consecutive onEpochWatermarkIncremented
     * invocations. 2) The coordinator operator has not observed any new value from any feedback
     * variable stream between two consecutive onEpochWatermarkIncremented invocations.
     *
     * <p>Required: 1) All the init variable streams and the data streams must be bounded. 2) The
     * parallelism of any stream in the initial variable streams must equal the parallelism of the
     * stream at the same index of the feedback variable streams returned by the IterationBody.
     *
     * @param initVariableStreams The initial variable streams. These streams will be merged with
     *     the feedback variable streams before being used as the 1st parameter to invoke the
     *     iteration body.
     * @param initDataStreams The initial data streams. Records from these streams will be
     *     repeatedly replayed and used as the 2nd parameter to invoke the iteration body.
     * @param body The computation logic which takes variable/data streams and returns
     *     variable/output streams.
     * @return The list of output streams returned by the iteration boy.
     */
    static DataStreamList iterateAndReplayBoundedStreamsUntilTermination(
            DataStreamList initVariableStreams,
            DataStreamList initDataStreams,
            IterationBody body) {
        return null;
    }
}
