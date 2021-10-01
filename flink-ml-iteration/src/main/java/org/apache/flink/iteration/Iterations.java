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
import org.apache.flink.iteration.operator.allround.AllRoundOperatorWrapper;
import org.apache.flink.util.Preconditions;

/**
 * A helper class to create iterations. To construct an iteration, Users are required to provide
 *
 * <ul>
 *   <li>initVariableStreams: the initial values of the variable data streams which would be updated
 *       in each round.
 *   <li>dataStreams: the other data streams used inside the iteration, but would not be updated.
 *   <li>iterationBody: specifies the subgraph to update the variable streams and the outputs.
 * </ul>
 *
 * <p>The iteration body will be invoked with two parameters: The first parameter is a list of input
 * variable streams, which are created as the union of the initial variable streams and the
 * corresponding feedback variable streams (returned by the iteration body); The second parameter is
 * the data streams given to this method.
 *
 * <p>During the execution of iteration body, each of the records involved in the iteration has an
 * epoch attached, which is mark the progress of the iteration. The epoch is computed as:
 *
 * <ul>
 *   <li>All records in the initial variable streams and initial data streams has epoch = 0.
 *   <li>For any record emitted by this operator into a non-feedback stream, the epoch of this
 *       emitted record = the epoch of the input record that triggers this emission. If this record
 *       is emitted by onEpochWatermarkIncremented(), then the epoch of this record =
 *       epochWatermark.
 *   <li>For any record emitted by this operator into a feedback variable stream, the epoch of the
 *       emitted record = the epoch of the input record that triggers this emission + 1.
 * </ul>
 *
 * <p>The framework would given the notification at the end of each epoch for operators and UDFs
 * that implements {@link IterationListener}.
 *
 * <p>The limitation of constructing the subgraph inside the iteration body could be refer in {@link
 * IterationBody}.
 *
 * <p>An example of the iteration is like:
 *
 * <pre>{@code
 * DataStreamList result = Iterations.iterateUnboundedStreams(
 *  DataStreamList.of(first, second),
 *  DataStreamList.of(third),
 *  (variableStreams, dataStreams) -> {
 *      ...
 *      return new IterationBodyResult(
 *          DataStreamList.of(firstFeedback, secondFeedback),
 *          DataStreamList.of(output));
 *  }
 *  result.<Integer>get(0).addSink(...);
 * }</pre>
 */
@Experimental
public class Iterations {

    /**
     * This method uses an iteration body to process records in possibly unbounded data streams. The
     * iteration would not terminate if at least one of its inputs is unbounded. Otherwise it will
     * terminated after all the inputs are terminated and no more records are iterating.
     *
     * @param initVariableStreams The initial variable streams, which is merged with the feedback
     *     variable streams before being used as the 1st parameter to invoke the iteration body.
     * @param dataStreams The non-variable streams also refer in the {@code body}.
     * @param body The computation logic which takes variable/data streams and returns
     *     feedback/output streams.
     * @return The list of output streams returned by the iteration boy.
     */
    public static DataStreamList iterateUnboundedStreams(
            DataStreamList initVariableStreams, DataStreamList dataStreams, IterationBody body) {
        return IterationFactory.createIteration(
                initVariableStreams, dataStreams, body, new AllRoundOperatorWrapper(), false);
    }

    /**
     * This method uses an iteration body to process records in some bounded data streams
     * iteratively until no more records are iterating or the given terminating criteria stream is
     * empty in one round.
     *
     * @param initVariableStreams The initial variable streams, which is merged with the feedback
     *     variable streams before being used as the 1st parameter to invoke the iteration body.
     * @param dataStreams The non-variable streams also refer in the {@code body} and if each of
     *     them needs replayed for each round.
     * @param config The config for the iteration, like whether to re-create the operator on each
     *     round.
     * @param body The computation logic which takes variable/data streams and returns
     *     feedback/output streams.
     * @return The list of output streams returned by the iteration boy.
     */
    public static DataStreamList iterateBoundedStreamsUntilTermination(
            DataStreamList initVariableStreams,
            ReplayableDataStreamList dataStreams,
            IterationConfig config,
            IterationBody body) {
        Preconditions.checkArgument(
                config.getOperatorLifeCycle() == IterationConfig.OperatorLifeCycle.ALL_ROUND);
        Preconditions.checkArgument(dataStreams.getReplayedDataStreams().size() == 0);

        return IterationFactory.createIteration(
                initVariableStreams,
                new DataStreamList(dataStreams.getNonReplayedStreams()),
                body,
                new AllRoundOperatorWrapper(),
                true);
    }
}
