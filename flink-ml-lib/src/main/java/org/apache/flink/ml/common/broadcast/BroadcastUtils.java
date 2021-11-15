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

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.iteration.compile.DraftExecutionEnvironment;
import org.apache.flink.ml.common.broadcast.operator.BroadcastVariableReceiverOperatorFactory;
import org.apache.flink.ml.common.broadcast.operator.BroadcastWrapper;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.MultipleConnectedStreams;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.ChainingStrategy;
import org.apache.flink.streaming.api.transformations.MultipleInputTransformation;
import org.apache.flink.streaming.api.transformations.PhysicalTransformation;
import org.apache.flink.util.AbstractID;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

/** Utility class to support withBroadcast in DataStream. */
@Internal
public class BroadcastUtils {
    /**
     * supports withBroadcastStream in DataStream API. Broadcast data streams are available at all
     * parallel instances of an operator that extends {@code
     * org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator<OUT, ? extends
     * org.apache.flink.api.common.functions.RichFunction>}. Users can access the broadcast
     * variables by {@code RichFunction.getRuntimeContext().getBroadcastVariable(...)} or {@code
     * RichFunction.getRuntimeContext().hasBroadcastVariable(...)} or {@code
     * RichFunction.getRuntimeContext().getBroadcastVariableWithInitializer(...)}.
     *
     * <p>In detail, the broadcast input data streams will be consumed first and further consumed by
     * non-broadcast inputs. For now the non-broadcast input are cached by default to avoid the
     * possible deadlocks.
     *
     * @param inputList non-broadcast input list.
     * @param bcStreams map of the broadcast data streams, where the key is the name and the value
     *     is the corresponding data stream.
     * @param userDefinedFunction the user defined logic in which users can access the broadcast
     *     data streams and produce the output data stream. Note that users can add only one
     *     operator in this function, otherwise it raises an exception.
     * @return the output data stream.
     */
    @Internal
    public static <OUT> DataStream<OUT> withBroadcastStream(
            List<DataStream<?>> inputList,
            Map<String, DataStream<?>> bcStreams,
            Function<List<DataStream<?>>, DataStream<OUT>> userDefinedFunction) {
        Preconditions.checkArgument(inputList.size() > 0);

        StreamExecutionEnvironment env = inputList.get(0).getExecutionEnvironment();
        String[] broadcastNames = new String[bcStreams.size()];
        DataStream<?>[] broadcastInputs = new DataStream[bcStreams.size()];
        TypeInformation<?>[] broadcastInTypes = new TypeInformation[bcStreams.size()];
        int idx = 0;
        final String broadcastId = new AbstractID().toHexString();
        for (String name : bcStreams.keySet()) {
            broadcastNames[idx] = broadcastId + "-" + name;
            broadcastInputs[idx] = bcStreams.get(name);
            broadcastInTypes[idx] = broadcastInputs[idx].getType();
            idx++;
        }

        DataStream<OUT> resultStream =
                getResultStream(env, inputList, broadcastNames, userDefinedFunction);
        TypeInformation<OUT> outType = resultStream.getType();
        final String coLocationKey = "broadcast-co-location-" + UUID.randomUUID();
        DataStream<OUT> cachedBroadcastInputs =
                cacheBroadcastVariables(
                        env,
                        broadcastNames,
                        broadcastInputs,
                        broadcastInTypes,
                        resultStream.getParallelism(),
                        outType);

        boolean canCoLocate =
                cachedBroadcastInputs.getTransformation() instanceof PhysicalTransformation
                        && resultStream.getTransformation() instanceof PhysicalTransformation;
        if (canCoLocate) {
            ((PhysicalTransformation<?>) cachedBroadcastInputs.getTransformation())
                    .setChainingStrategy(ChainingStrategy.HEAD);
            ((PhysicalTransformation<?>) resultStream.getTransformation())
                    .setChainingStrategy(ChainingStrategy.HEAD);
        } else {
            throw new UnsupportedOperationException(
                    "cannot set chaining strategy on "
                            + cachedBroadcastInputs.getTransformation()
                            + " and "
                            + resultStream.getTransformation()
                            + ".");
        }
        cachedBroadcastInputs.getTransformation().setCoLocationGroupKey(coLocationKey);
        resultStream.getTransformation().setCoLocationGroupKey(coLocationKey);

        return cachedBroadcastInputs.union(resultStream);
    }

    /**
     * caches all broadcast iput data streams in static variables and returns the result multi-input
     * stream operator. The result multi-input stream operator emits nothing and the only
     * functionality of this operator is to cache all the input records in ${@link
     * BroadcastContext}.
     *
     * @param env execution environment.
     * @param broadcastInputNames names of the broadcast input data streams.
     * @param broadcastInputs list of the broadcast data streams.
     * @param broadcastInTypes output types of the broadcast input data streams.
     * @param parallelism parallelism.
     * @param outType output type.
     * @param <OUT> output type.
     * @return the result multi-input stream operator.
     */
    private static <OUT> DataStream<OUT> cacheBroadcastVariables(
            StreamExecutionEnvironment env,
            String[] broadcastInputNames,
            DataStream<?>[] broadcastInputs,
            TypeInformation<?>[] broadcastInTypes,
            int parallelism,
            TypeInformation<OUT> outType) {
        MultipleInputTransformation<OUT> transformation =
                new MultipleInputTransformation<>(
                        "broadcastInputs",
                        new BroadcastVariableReceiverOperatorFactory<>(
                                broadcastInputNames, broadcastInTypes),
                        outType,
                        parallelism);
        for (DataStream<?> dataStream : broadcastInputs) {
            transformation.addInput(dataStream.broadcast().getTransformation());
        }
        env.addOperator(transformation);
        return new MultipleConnectedStreams(env).transform(transformation);
    }

    /**
     * uses {@link DraftExecutionEnvironment} to execute the userDefinedFunction and returns the
     * resultStream.
     *
     * @param env execution environment.
     * @param inputList non-broadcast input list.
     * @param broadcastStreamNames names of the broadcast data streams.
     * @param graphBuilder user-defined logic.
     * @param <OUT> output type of the result stream.
     * @return the result stream by applying user-defined logic on the input list.
     */
    private static <OUT> DataStream<OUT> getResultStream(
            StreamExecutionEnvironment env,
            List<DataStream<?>> inputList,
            String[] broadcastStreamNames,
            Function<List<DataStream<?>>, DataStream<OUT>> graphBuilder) {
        TypeInformation<?>[] inTypes = new TypeInformation[inputList.size()];
        for (int i = 0; i < inputList.size(); i++) {
            inTypes[i] = inputList.get(i).getType();
        }
        // do not block all non-broadcast input edges by default.
        boolean[] isBlocked = new boolean[inputList.size()];
        Arrays.fill(isBlocked, false);
        DraftExecutionEnvironment draftEnv =
                new DraftExecutionEnvironment(
                        env, new BroadcastWrapper<>(broadcastStreamNames, inTypes, isBlocked));

        List<DataStream<?>> draftSources = new ArrayList<>();
        for (DataStream<?> dataStream : inputList) {
            draftSources.add(draftEnv.addDraftSource(dataStream, dataStream.getType()));
        }
        DataStream<OUT> draftOutStream = graphBuilder.apply(draftSources);
        Preconditions.checkState(
                draftEnv.getStreamGraph(false).getStreamNodes().size() == 1 + inputList.size(),
                "cannot add more than one operator in withBroadcastStream's lambda function.");
        draftEnv.copyToActualEnvironment();
        return draftEnv.getActualStream(draftOutStream.getId());
    }
}
