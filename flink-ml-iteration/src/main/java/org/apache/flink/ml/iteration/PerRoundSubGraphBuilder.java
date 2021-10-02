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

import org.apache.flink.annotation.Internal;
import org.apache.flink.ml.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.ml.iteration.operator.OperatorWrapper;
import org.apache.flink.ml.iteration.operator.perrond.PerRoundOperatorWrapper;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.util.List;
import java.util.Optional;

import static org.apache.flink.util.Preconditions.checkArgument;

/** Allows to add per-round subgraph inside the iteration body. */
@Internal
public class PerRoundSubGraphBuilder {

    public interface PerRoundSubGraph {

        List<DataStreamList> process(List<DataStreamList> input);
    }

    public static List<DataStreamList> forEachRound(
            List<DataStreamList> inputs, PerRoundSubGraph subGraph) {
        Optional<DataStream<?>> first =
                inputs.stream().flatMap(input -> input.getDataStreams().stream()).findFirst();
        checkArgument(first.isPresent(), "At least one input is required");
        DraftExecutionEnvironment env =
                (DraftExecutionEnvironment) first.get().getExecutionEnvironment();

        ensureExplicitlyAdded(env, inputs);
        OperatorWrapper<?, ?> oldWrapper = env.setCurrentWrapper(new PerRoundOperatorWrapper<>());
        List<DataStreamList> outputs = subGraph.process(inputs);
        ensureExplicitlyAdded(env, outputs);
        env.setCurrentWrapper(oldWrapper);

        return outputs;
    }

    private static void ensureExplicitlyAdded(
            DraftExecutionEnvironment env, List<DataStreamList> dataStreamLists) {
        for (int i = 0; i < dataStreamLists.size(); ++i) {
            for (int j = 0; j < dataStreamLists.get(i).size(); ++j) {
                env.addOperatorIfNotExists(dataStreamLists.get(i).get(j).getTransformation());
            }
        }
    }
}
