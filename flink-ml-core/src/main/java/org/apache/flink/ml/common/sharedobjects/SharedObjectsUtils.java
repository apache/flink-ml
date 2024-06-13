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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.api.dag.Transformation;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

/** Utility class to support shared objects mechanism in DataStream. */
@Experimental
public class SharedObjectsUtils {

    /**
     * Supports read/write access of data in the shared objects from operators which inherit {@link
     * AbstractSharedObjectsStreamOperator}.
     *
     * <p>In the shared objects `body`, users build the subgraph with data streams only from
     * `inputs`, return streams that have access to the shared objects, and return the mapping from
     * shared objects to their owners.
     *
     * <p>There are several limitations to use this function:
     *
     * <ol>
     *   <li>Only synchronized iterations and non-iterations are supported.
     *   <li>Reads and writes of shared objects must obey strict rules defined on `step`s, as stated
     *       in {@link ReadRequest}.
     *   <li>When in iterations, writes of shared objects can only occur in {@link
     *       IterationListener#onEpochWatermarkIncremented} and {@link
     *       IterationListener#onIterationTerminated}.
     * </ol>
     *
     * @param inputs Input data streams.
     * @param body User defined logic to build subgraph and to specify owners of every shared
     *     object.
     * @return The output data streams.
     */
    public static List<DataStream<?>> withSharedObjects(
            List<DataStream<?>> inputs, SharedObjectsBody body) {
        Preconditions.checkArgument(!inputs.isEmpty());
        StreamExecutionEnvironment env = inputs.get(0).getExecutionEnvironment();
        String coLocationID = "shared-storage-" + UUID.randomUUID();
        SharedObjectsContextImpl context = new SharedObjectsContextImpl();

        DraftExecutionEnvironment draftEnv =
                new DraftExecutionEnvironment(env, new SharedObjectsWrapper<>(context));
        List<DataStream<?>> draftSources =
                inputs.stream()
                        .map(
                                dataStream ->
                                        draftEnv.addDraftSource(dataStream, dataStream.getType()))
                        .collect(Collectors.toList());
        SharedObjectsBody.SharedObjectsBodyResult result = body.process(draftSources);

        List<DataStream<?>> draftOutputs = result.getOutputs();
        Map<Descriptor<?>, AbstractSharedObjectsStreamOperator<?>> rawOwnerMap =
                result.getOwnerMap();
        Map<Descriptor<?>, String> ownerMap = new HashMap<>();
        for (Descriptor<?> descriptor : rawOwnerMap.keySet()) {
            ownerMap.put(descriptor, rawOwnerMap.get(descriptor).getAccessorID());
        }
        context.setOwnerMap(ownerMap);

        for (DataStream<?> draftOutput : draftOutputs) {
            draftEnv.addOperator(draftOutput.getTransformation());
        }
        draftEnv.copyToActualEnvironment();

        for (Transformation<?> transformation : result.getCoLocatedTransformations()) {
            DataStream<?> ds = draftEnv.getActualStream(transformation.getId());
            ds.getTransformation().setCoLocationGroupKey(coLocationID);
        }

        List<DataStream<?>> outputs = new ArrayList<>();
        for (DataStream<?> draftOutput : draftOutputs) {
            outputs.add(draftEnv.getActualStream(draftOutput.getId()));
        }
        return outputs;
    }
}
