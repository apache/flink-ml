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

package org.apache.flink.ml.common.ps.training;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.Partitioner;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.ps.ResponseAssemblerOperator;
import org.apache.flink.ml.common.ps.ServerOperator;
import org.apache.flink.ml.common.ps.WorkerOperator;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.ml.util.Bits;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.util.OutputTag;

import java.util.ArrayList;
import java.util.List;

/** Utility function to describe iterative training process. */
public final class TrainingUtils {
    /**
     * Executes the training logic described in {@link IterationStageList} and returns the fitted
     * model data as well as the outputs from worker operator. The outputs from worker operator are
     * specified via {@link MLSession#getOutputTags()}.
     *
     * @param inputData the input data.
     * @param iterationStages the iterative processing logic.
     * @param maxKey max value of the key. For example, the maxKey should be the max feature index
     *     in LogisticRegression.
     * @param modelDataType output type information of model data.
     * @param modelUpdater the logic to update model on servers.
     * @param numServers number of servers.
     * @return the fitted model data as well as the outputs from worker operator. The orders are
     *     {modelData, sideOutputs from workers}. Note that the outputs from workers shares the same
     *     order with the {@link MLSession#getOutputTags()}.
     * @param <DT> type information of input data.
     * @param <MT> type information of the output model data.
     */
    public static <DT, MT> DataStreamList train(
            DataStream<DT> inputData,
            IterationStageList<? extends MLSession> iterationStages,
            DataStream<Long> maxKey,
            TypeInformation<MT> modelDataType,
            ModelUpdater<MT> modelUpdater,
            int numServers) {
        // TODO: Support incremental training.

        DataStream<byte[]> variableStream =
                maxKey.broadcast()
                        .map(
                                (MapFunction<Long, byte[]>)
                                        value -> {
                                            byte[] buffer = new byte[Long.BYTES];
                                            Bits.putLong(buffer, 0, value + 1);
                                            return buffer;
                                        });

        return Iterations.iterateBoundedStreamsUntilTermination(
                DataStreamList.of(variableStream),
                ReplayableDataStreamList.notReplay(inputData),
                IterationConfig.newBuilder().build(),
                new TrainIterationBody<>(modelUpdater, modelDataType, iterationStages, numServers));
    }

    /** The iteration implementation for training process. */
    private static class TrainIterationBody<MT> implements IterationBody {
        private final ModelUpdater<MT> modelUpdater;

        private final TypeInformation<MT> modelType;
        private final IterationStageList<? extends MLSession> iterationStages;
        private final int numServers;

        public TrainIterationBody(
                ModelUpdater<MT> modelUpdater,
                TypeInformation<MT> modelType,
                IterationStageList<? extends MLSession> iterationStages,
                int numServers) {
            this.iterationStages = iterationStages;
            this.modelType = modelType;
            this.modelUpdater = modelUpdater;
            this.numServers = numServers;
        }

        @Override
        @SuppressWarnings("unchecked")
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<byte[]> variableStream = variableStreams.get(0);
            DataStream<LabeledPointWithWeight> trainData = dataStreams.get(0);
            final OutputTag<MT> modelDataOutputTag = new OutputTag<>("MODEL_OUTPUT", modelType);

            SingleOutputStreamOperator<Tuple2<Integer, byte[]>> messageToServer =
                    trainData
                            .connect(variableStream)
                            .transform(
                                    "workerNode",
                                    new TupleTypeInfo<>(
                                            Types.INT,
                                            PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO),
                                    new WorkerOperator(iterationStages, numServers))
                            .name("WorkerOp");
            int numWorkers = messageToServer.getParallelism();

            SingleOutputStreamOperator<Tuple2<Integer, byte[]>> messageToWorker =
                    messageToServer
                            .partitionCustom(
                                    (Partitioner<Integer>)
                                            (key, numPartitions) -> key % numPartitions,
                                    (KeySelector<Tuple2<Integer, byte[]>, Integer>)
                                            value -> value.f0)
                            .transform(
                                    "ServerOp",
                                    new TupleTypeInfo<>(
                                            Types.INT,
                                            PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO),
                                    new ServerOperator<>(
                                            iterationStages,
                                            numWorkers,
                                            modelUpdater,
                                            modelDataOutputTag));
            messageToWorker.setParallelism(numServers);

            DataStream<byte[]> combinedMessageToWorker =
                    messageToWorker
                            .partitionCustom(
                                    (Partitioner<Integer>)
                                            (key, numPartitions) -> key % numPartitions,
                                    (KeySelector<Tuple2<Integer, byte[]>, Integer>)
                                            value -> value.f0)
                            .transform(
                                    "MirrorWorkerOp",
                                    PrimitiveArrayTypeInfo.BYTE_PRIMITIVE_ARRAY_TYPE_INFO,
                                    new ResponseAssemblerOperator(numServers))
                            .setParallelism(numWorkers);

            DataStream<MT> model = messageToWorker.getSideOutput(modelDataOutputTag);

            List<DataStream<?>> result = new ArrayList<>();
            result.add(model);

            List<OutputTag<?>> sideOutputTags = iterationStages.session.getOutputTags();
            if (sideOutputTags != null) {
                for (OutputTag<?> outputTag : sideOutputTags) {
                    result.add(messageToServer.getSideOutput(outputTag));
                }
            }

            return new IterationBodyResult(
                    DataStreamList.of(combinedMessageToWorker), new DataStreamList(result), null);
        }
    }
}
