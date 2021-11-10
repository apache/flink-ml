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
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.iteration.operator.OperatorWrapper;
import org.apache.flink.iteration.operator.perround.PerRoundOperatorWrapper;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.io.Serializable;

import static org.apache.flink.util.Preconditions.checkArgument;

/**
 * The builder of the subgraph that will be executed inside the iteration.
 *
 * <p>Notes that inside the iteration body, users could only create the subgraph from the {@code
 * variableStreams} and {@code dataStreams}. Users could not refers to other data stream outside the
 * iteration through the closure, and could not add new sources / sinks inside the iteration.
 *
 * <p>Some operations are not supported inside the iterations:
 *
 * <ul>
 *   <li>Sources and Sinks.
 *   <li>{@link DataStream#assignTimestampsAndWatermarks(WatermarkStrategy)}.
 *   <li>{@link DataStream#iterate()}.
 * </ul>
 *
 * <p>Currently we also not support nested exception.
 *
 * <p>The iteration body also requires that the parallelism of any stream in the initial variable
 * streams must equal to the parallelism of the stream at the same index of the feedback variable
 * streams returned by the iteration body.
 */
@Experimental
public interface IterationBody extends Serializable {

    /**
     * This method creates the graph for the iteration body. See {@link Iterations} for how the
     * iteration body can be executed and terminated.
     *
     * @param variableStreams the variable streams, which will be updated via a feedback stream in
     *     each round.
     * @param dataStreams the streams referred in the iteration body, which will only be emitted in
     *     the first round.
     * @return the result of the iteration, including the feedbacks and outputs.
     */
    IterationBodyResult process(DataStreamList variableStreams, DataStreamList dataStreams);

    /**
     * @param inputs The inputs of the subgraph.
     * @param perRoundSubBody The computational logic that want to be executed as per-round.
     * @return The output of the subgraph.
     */
    static DataStreamList forEachRound(DataStreamList inputs, PerRoundSubBody perRoundSubBody) {
        checkArgument(inputs.size() > 0, "At least one input is required");
        DataStream<?> first = inputs.get(0);
        DraftExecutionEnvironment env = (DraftExecutionEnvironment) first.getExecutionEnvironment();

        for (int i = 0; i < inputs.size(); ++i) {
            env.addOperatorIfNotExists(inputs.get(i).getTransformation());
        }
        OperatorWrapper<?, ?> oldWrapper = env.setCurrentWrapper(new PerRoundOperatorWrapper<>());

        DataStreamList outputs = perRoundSubBody.process(inputs);
        for (int i = 0; i < inputs.size(); ++i) {
            env.addOperatorIfNotExists(inputs.get(i).getTransformation());
        }

        env.setCurrentWrapper(oldWrapper);

        return outputs;
    }

    /** The sub-graph inside the iteration body that should be executed as per-round. */
    interface PerRoundSubBody {

        DataStreamList process(DataStreamList input);
    }
}
