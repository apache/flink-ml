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

package org.apache.flink.ml.common.sharedstorage;

import org.apache.flink.annotation.Experimental;
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

/** Utility class to support {@link SharedStorage} in DataStream. */
@Experimental
public class SharedStorageUtils {

    /**
     * Support read/write access of data in the shared storage from operators which implements
     * {@link SharedStorageStreamOperator}.
     *
     * <p>In the shared storage `body`, users build the subgraph with data streams only from
     * `inputs`, return streams that have access to the shared storage, and return the mapping from
     * shared items to their owners.
     *
     * @param inputs Input data streams.
     * @param body User defined logic to build subgraph and to specify owners of every shared data
     *     item.
     * @return The output data streams.
     */
    public static List<DataStream<?>> withSharedStorage(
            List<DataStream<?>> inputs, SharedStorageBody body) {
        Preconditions.checkArgument(inputs.size() > 0);
        StreamExecutionEnvironment env = inputs.get(0).getExecutionEnvironment();
        String coLocationID = "shared-storage-" + UUID.randomUUID();
        SharedStorageContextImpl context = new SharedStorageContextImpl();

        DraftExecutionEnvironment draftEnv =
                new DraftExecutionEnvironment(env, new SharedStorageWrapper<>(context));
        List<DataStream<?>> draftSources =
                inputs.stream()
                        .map(
                                dataStream ->
                                        draftEnv.addDraftSource(dataStream, dataStream.getType()))
                        .collect(Collectors.toList());
        SharedStorageBody.SharedStorageBodyResult result = body.process(draftSources);

        List<DataStream<?>> draftOutputs = result.getOutputs();
        Map<ItemDescriptor<?>, SharedStorageStreamOperator> rawOwnerMap = result.getOwnerMap();
        Map<ItemDescriptor<?>, String> ownerMap = new HashMap<>();
        for (ItemDescriptor<?> item : rawOwnerMap.keySet()) {
            ownerMap.put(item, rawOwnerMap.get(item).getSharedStorageAccessorID());
        }
        context.setOwnerMap(ownerMap);

        for (DataStream<?> draftOutput : draftOutputs) {
            draftEnv.addOperator(draftOutput.getTransformation());
        }
        draftEnv.copyToActualEnvironment();

        for (DataStream<?> accessor : result.getAccessors()) {
            DataStream<?> ds = draftEnv.getActualStream(accessor.getTransformation().getId());
            ds.getTransformation().setCoLocationGroupKey(coLocationID);
        }

        List<DataStream<?>> outputs = new ArrayList<>();
        for (DataStream<?> draftOutput : draftOutputs) {
            outputs.add(draftEnv.getActualStream(draftOutput.getId()));
        }
        return outputs;
    }
}
