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

package org.apache.flink.ml.common.broadcast;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.ml.common.broadcast.operator.BroadcastWrapper;
import org.apache.flink.ml.common.broadcast.operator.CacheStreamOperatorFactory;
import org.apache.flink.ml.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.MultipleConnectedStreams;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.transformations.MultipleInputTransformation;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class BroadcastUtils {

    private static <OUT> DataStream<OUT> cacheBroadcastVariables(
            StreamExecutionEnvironment env,
            Map<String, DataStream<?>> bcStreams,
            TypeInformation<OUT> outType) {
        int numBroadcastInput = bcStreams.size();
        String[] broadcastInputNames = bcStreams.keySet().toArray(new String[0]);
        DataStream<?>[] broadcastInputs = bcStreams.values().toArray(new DataStream<?>[0]);
        TypeInformation<?>[] broadcastInTypes = new TypeInformation[numBroadcastInput];
        for (int i = 0; i < numBroadcastInput; i++) {
            broadcastInTypes[i] = broadcastInputs[i].getType();
        }

        MultipleInputTransformation<OUT> transformation =
                new MultipleInputTransformation<OUT>(
                        "broadcastInputs",
                        new CacheStreamOperatorFactory<OUT>(broadcastInputNames, broadcastInTypes),
                        outType,
                        env.getParallelism());
        for (DataStream<?> dataStream : bcStreams.values()) {
            transformation.addInput(dataStream.broadcast().getTransformation());
        }
        env.addOperator(transformation);
        return new MultipleConnectedStreams(env).transform(transformation);
    }

    private static String getCoLocationKey(String[] broadcastNames) {
        StringBuilder sb = new StringBuilder();
        sb.append("Flink-ML-broadcast-co-location");
        for (String name : broadcastNames) {
            sb.append(name);
        }
        return sb.toString();
    }

    private static <OUT> DataStream<OUT> buildGraph(
            StreamExecutionEnvironment env,
            List<DataStream<?>> inputList,
            String[] broadcastStreamNames,
            Function<List<DataStream<?>>, DataStream<OUT>> graphBuilder) {
        TypeInformation[] inTypes = new TypeInformation[inputList.size()];
        for (int i = 0; i < inputList.size(); i++) {
            TypeInformation type = inputList.get(i).getType();
            inTypes[i] = type;
        }
        // blocking all non-broadcast input edges by default.
        boolean[] isBlocking = new boolean[inTypes.length];
        Arrays.fill(isBlocking, true);
        DraftExecutionEnvironment draftEnv =
                new DraftExecutionEnvironment(
                        env, new BroadcastWrapper<>(broadcastStreamNames, inTypes, isBlocking));

        List<DataStream<?>> draftSources = new ArrayList<>();
        for (int i = 0; i < inputList.size(); i++) {
            draftSources.add(draftEnv.addDraftSource(inputList.get(i), inputList.get(i).getType()));
        }
        DataStream<OUT> draftOutStream = graphBuilder.apply(draftSources);

        draftEnv.copyToActualEnvironment();
        DataStream<OUT> outStream = draftEnv.getActualStream(draftOutStream.getId());
        return outStream;
    }

    /**
     * Support withBroadcastStream in DataStream API. Broadcast data streams are available at all
     * parallel instances of the input operators. A broadcast data stream is registered under a
     * certain name and can be retrieved under that name via {@link
     * BroadcastContext}.getBroadcastVariable(...).
     *
     * <p>In detail, the broadcast input data streams will be consumed first and cached as static
     * variables in {@link BroadcastContext}. For now the non-broadcast input are blocking and
     * cached to avoid the possible deadlocks.
     *
     * @param inputList the non-broadcast input list.
     * @param bcStreams map of the broadcast data streams, where the key is the name and the value
     *     is the corresponding data stream.
     * @param userDefinedFunction the user defined logic in which users can access the broadcast
     *     data streams and produce the output data stream.
     * @param <OUT> type of the output data stream.
     * @return the output data stream.
     */
    @PublicEvolving
    public static <OUT> DataStream<OUT> withBroadcastStream(
            List<DataStream<?>> inputList,
            Map<String, DataStream<?>> bcStreams,
            Function<List<DataStream<?>>, DataStream<OUT>> userDefinedFunction) {
        Preconditions.checkState(inputList.size() > 0);
        StreamExecutionEnvironment env = inputList.get(0).getExecutionEnvironment();
        final String[] broadcastStreamNames = bcStreams.keySet().toArray(new String[0]);
        DataStream<OUT> resultStream =
                buildGraph(env, inputList, broadcastStreamNames, userDefinedFunction);

        TypeInformation outType = resultStream.getType();
        final String coLocationKey = getCoLocationKey(broadcastStreamNames);
        DataStream<OUT> cachedBroadcastInputs = cacheBroadcastVariables(env, bcStreams, outType);

        for (int i = 0; i < inputList.size(); i++) {
            inputList.get(i).getTransformation().setCoLocationGroupKey(coLocationKey);
        }
        cachedBroadcastInputs.getTransformation().setCoLocationGroupKey(coLocationKey);

        return cachedBroadcastInputs.union(resultStream);
    }
}
