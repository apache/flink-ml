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
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.ps.MirrorWorkerOperator;
import org.apache.flink.ml.common.ps.ServerOperator;
import org.apache.flink.ml.common.ps.WorkerOperator;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.ml.util.Bits;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.util.OutputTag;

/** Utility function to describe iterative training process. */
public final class TrainingUtils {

    /**
     * Executes the training logic described in {@link IterationStageList} and returns the fitted
     * model data.
     *
     * @param modelDim dimension of the input model.
     * @param trainData the training data.
     * @param iterationStages the iterative training logic.
     * @param modelUpdater the logic to update model on servers.
     * @param numServers number of servers.
     * @return the fitted model data.
     */
    public static <T> DataStream<Tuple3<Long, Long, double[]>> train(
            DataStream<Long> modelDim,
            DataStream<T> trainData,
            ModelUpdater modelUpdater,
            IterationStageList<? extends MLSession> iterationStages,
            int numServers) {
        // TODO: Support incremental training for multiple models.

        DataStream<byte[]> variableStream =
                modelDim.broadcast()
                        .map(
                                (MapFunction<Long, byte[]>)
                                        value -> {
                                            byte[] buffer = new byte[Long.BYTES];
                                            Bits.putLong(buffer, 0, value);
                                            return buffer;
                                        });

        DataStreamList resultList =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(variableStream),
                        ReplayableDataStreamList.notReplay(
                                trainData.rebalance().map(x -> x, trainData.getType())),
                        IterationConfig.newBuilder().build(),
                        new TrainIterationBody(modelUpdater, iterationStages, numServers));

        return resultList.get(0);
    }

    /** The iteration implementation for training process. */
    private static class TrainIterationBody implements IterationBody {
        private final ModelUpdater modelUpdater;
        private final IterationStageList<? extends MLSession> iterationStages;
        private final int numServers;

        public TrainIterationBody(
                ModelUpdater modelUpdater,
                IterationStageList<? extends MLSession> iterationStages,
                int numServers) {
            this.iterationStages = iterationStages;
            this.modelUpdater = modelUpdater;
            this.numServers = numServers;
        }

        @Override
        @SuppressWarnings("unchecked")
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<byte[]> variableStream = variableStreams.get(0);
            DataStream<LabeledPointWithWeight> trainData = dataStreams.get(0);
            final OutputTag<Tuple3<Long, Long, double[]>> modelDataOutputTag =
                    new OutputTag<Tuple3<Long, Long, double[]>>("MODEL_OUTPUT") {};

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
                                    new ServerOperator(
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
                                    new MirrorWorkerOperator(numServers))
                            .setParallelism(numWorkers);

            return new IterationBodyResult(
                    DataStreamList.of(combinedMessageToWorker),
                    DataStreamList.of(messageToWorker.getSideOutput(modelDataOutputTag)),
                    null);
        }
    }
}
