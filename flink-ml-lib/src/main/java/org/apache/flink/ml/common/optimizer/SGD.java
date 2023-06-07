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

package org.apache.flink.ml.common.optimizer;

import org.apache.flink.annotation.Internal;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIterOrTol;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.regression.linearregression.LinearRegression;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import org.apache.commons.collections.IteratorUtils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Stochastic Gradient Descent (SGD) is the mostly wide-used optimizer for optimizing machine
 * learning models. It iteratively makes small adjustments to the machine learning model according
 * to the gradient at each step, to decrease the error of the model.
 *
 * <p>See https://en.wikipedia.org/wiki/Stochastic_gradient_descent.
 */
@Internal
public class SGD implements Optimizer {
    /** Params for SGD optimizer. */
    private final SGDParams params;

    public SGD(
            int maxIter,
            double learningRate,
            int globalBatchSize,
            double tol,
            double reg,
            double elasticNet) {
        this.params = new SGDParams(maxIter, learningRate, globalBatchSize, tol, reg, elasticNet);
    }

    @Override
    public DataStream<DenseIntDoubleVector> optimize(
            DataStream<DenseIntDoubleVector> initModelData,
            DataStream<LabeledPointWithWeight> trainData,
            LossFunc lossFunc) {
        DataStreamList resultList =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(
                                initModelData.broadcast().map(modelVec -> modelVec.values)),
                        ReplayableDataStreamList.notReplay(trainData.rebalance().map(x -> x)),
                        IterationConfig.newBuilder().build(),
                        new TrainIterationBody(lossFunc, params));
        return resultList.get(0);
    }

    /** The iteration implementation for training process. */
    private static class TrainIterationBody implements IterationBody {
        private final LossFunc lossFunc;
        private final SGDParams params;

        public TrainIterationBody(LossFunc lossFunc, SGDParams params) {
            this.lossFunc = lossFunc;
            this.params = params;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            // The variable stream at the first iteration is the initialized model data.
            // In the following iterations, it contains: [the model update, totalWeight, and
            // totalLoss].
            DataStream<double[]> variableStream = variableStreams.get(0);
            DataStream<LabeledPointWithWeight> trainData = dataStreams.get(0);
            final OutputTag<DenseIntDoubleVector> modelDataOutputTag =
                    new OutputTag<DenseIntDoubleVector>("MODEL_OUTPUT") {};

            SingleOutputStreamOperator<double[]> modelUpdateAndWeightAndLoss =
                    trainData
                            .connect(variableStream)
                            .transform(
                                    "CacheDataAndDoTrain",
                                    PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO,
                                    new CacheDataAndDoTrain(lossFunc, params, modelDataOutputTag));

            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            DataStreamList.of(modelUpdateAndWeightAndLoss),
                            input -> {
                                DataStream<double[]> feedback =
                                        DataStreamUtils.allReduceSum(input.get(0));
                                return DataStreamList.of(feedback);
                            });

            DataStream<Integer> terminationCriteria =
                    feedbackVariableStream
                            .get(0)
                            .map(
                                    reducedUpdateAndWeightAndLoss -> {
                                        double[] value = (double[]) reducedUpdateAndWeightAndLoss;
                                        return value[value.length - 1] / value[value.length - 2];
                                    })
                            .flatMap(new TerminateOnMaxIterOrTol(params.maxIter, params.tol));

            return new IterationBodyResult(
                    DataStreamList.of(feedbackVariableStream.get(0)),
                    DataStreamList.of(
                            modelUpdateAndWeightAndLoss.getSideOutput(modelDataOutputTag)),
                    terminationCriteria);
        }
    }

    /**
     * A stream operator that caches the training data in the first iteration and updates the model
     * iteratively. The first input is the training data, and the second input is the initial model
     * data or feedback of model update, totalWeight, and totalLoss.
     */
    private static class CacheDataAndDoTrain extends AbstractStreamOperator<double[]>
            implements TwoInputStreamOperator<LabeledPointWithWeight, double[], double[]>,
                    IterationListener<double[]> {
        /** Optimizer-related parameters. */
        private final SGDParams params;

        /** The loss function to optimize. */
        private final LossFunc lossFunc;

        /** The outputTag to output the model data when iteration ends. */
        private final OutputTag<DenseIntDoubleVector> modelDataOutputTag;

        /** The cached training data. */
        private List<LabeledPointWithWeight> trainData;

        private ListState<LabeledPointWithWeight> trainDataState;

        /** The start index (offset) of the next mini-batch data for training. */
        private int nextBatchOffset = 0;

        private ListState<Integer> nextBatchOffsetState;

        /** The model coefficient. */
        private DenseIntDoubleVector coefficient;

        private ListState<DenseIntDoubleVector> coefficientState;

        /** The dimension of the coefficient. */
        private int coefficientDim;

        /**
         * The double array to sync among all workers. For example, when training {@link
         * LinearRegression}, this double array consists of [modelUpdate, totalWeight, totalLoss].
         */
        private double[] feedbackArray;

        private ListState<double[]> feedbackArrayState;

        /** The batch size on this partition. */
        private int localBatchSize;

        private CacheDataAndDoTrain(
                LossFunc lossFunc,
                SGDParams params,
                OutputTag<DenseIntDoubleVector> modelDataOutputTag) {
            this.lossFunc = lossFunc;
            this.params = params;
            this.modelDataOutputTag = modelDataOutputTag;
        }

        @Override
        public void open() {
            int numTasks = getRuntimeContext().getNumberOfParallelSubtasks();
            int taskId = getRuntimeContext().getIndexOfThisSubtask();
            localBatchSize = params.globalBatchSize / numTasks;
            if (params.globalBatchSize % numTasks > taskId) {
                localBatchSize++;
            }
        }

        private double getTotalWeight() {
            return feedbackArray[coefficientDim];
        }

        private void setTotalWeight(double totalWeight) {
            feedbackArray[coefficientDim] = totalWeight;
        }

        private double getTotalLoss() {
            return feedbackArray[coefficientDim + 1];
        }

        private void setTotalLoss(double totalLoss) {
            feedbackArray[coefficientDim + 1] = totalLoss;
        }

        private void updateModel() {
            if (getTotalWeight() > 0) {
                BLAS.axpy(
                        -params.learningRate / getTotalWeight(),
                        new DenseIntDoubleVector(feedbackArray),
                        coefficient,
                        coefficientDim);
                double regLoss =
                        RegularizationUtils.regularize(
                                coefficient, params.reg, params.elasticNet, params.learningRate);
                setTotalLoss(getTotalLoss() + regLoss);
            }
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<double[]> collector)
                throws Exception {
            if (epochWatermark == 0) {
                coefficient = new DenseIntDoubleVector(feedbackArray);
                coefficientDim = coefficient.size();
                feedbackArray = new double[coefficient.size() + 2];
            } else {
                updateModel();
            }

            if (trainData == null) {
                trainData = IteratorUtils.toList(trainDataState.get().iterator());
            }

            // TODO: supports efficient shuffle of training set on each partition.
            if (trainData.size() > 0) {
                List<LabeledPointWithWeight> miniBatchData =
                        trainData.subList(
                                nextBatchOffset,
                                Math.min(nextBatchOffset + localBatchSize, trainData.size()));
                nextBatchOffset += localBatchSize;
                nextBatchOffset = nextBatchOffset >= trainData.size() ? 0 : nextBatchOffset;

                // Does the training.
                Arrays.fill(feedbackArray, 0);
                double totalLoss = 0;
                double totalWeight = 0;
                DenseIntDoubleVector cumGradientsWrapper = new DenseIntDoubleVector(feedbackArray);
                for (LabeledPointWithWeight dataPoint : miniBatchData) {
                    totalLoss += lossFunc.computeLoss(dataPoint, coefficient);
                    lossFunc.computeGradient(dataPoint, coefficient, cumGradientsWrapper);
                    totalWeight += dataPoint.weight;
                }
                setTotalLoss(totalLoss);
                setTotalWeight(totalWeight);

                collector.collect(feedbackArray);
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<double[]> collector) {
            trainDataState.clear();
            if (getRuntimeContext().getIndexOfThisSubtask() == 0) {
                updateModel();
                context.output(modelDataOutputTag, coefficient);
            }
        }

        @Override
        public void processElement1(StreamRecord<LabeledPointWithWeight> streamRecord)
                throws Exception {
            trainDataState.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<double[]> streamRecord) {
            feedbackArray = streamRecord.getValue();
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            coefficientState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "coefficientState",
                                            DenseIntDoubleVectorTypeInfo.INSTANCE));
            OperatorStateUtils.getUniqueElement(coefficientState, "coefficientState")
                    .ifPresent(x -> coefficient = x);
            if (coefficient != null) {
                coefficientDim = coefficient.size();
            }

            feedbackArrayState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "feedbackArrayState",
                                            PrimitiveArrayTypeInfo
                                                    .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
            OperatorStateUtils.getUniqueElement(feedbackArrayState, "feedbackArrayState")
                    .ifPresent(x -> feedbackArray = x);

            trainDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "trainDataState",
                                            TypeInformation.of(LabeledPointWithWeight.class)));

            nextBatchOffsetState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "nextBatchOffsetState", BasicTypeInfo.INT_TYPE_INFO));
            nextBatchOffset =
                    OperatorStateUtils.getUniqueElement(
                                    nextBatchOffsetState, "nextBatchOffsetState")
                            .orElse(0);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            coefficientState.clear();
            if (coefficient != null) {
                coefficientState.add(coefficient);
            }

            feedbackArrayState.clear();
            if (feedbackArray != null) {
                feedbackArrayState.add(feedbackArray);
            }

            nextBatchOffsetState.clear();
            nextBatchOffsetState.add(nextBatchOffset);
        }
    }

    /** Parameters for {@link SGD}. */
    private static class SGDParams implements Serializable {
        public final int maxIter;
        public final double learningRate;
        public final int globalBatchSize;
        public final double tol;
        public final double reg;
        public final double elasticNet;

        private SGDParams(
                int maxIter,
                double learningRate,
                int globalBatchSize,
                double tol,
                double reg,
                double elasticNet) {
            this.maxIter = maxIter;
            this.learningRate = learningRate;
            this.globalBatchSize = globalBatchSize;
            this.tol = tol;
            this.reg = reg;
            this.elasticNet = elasticNet;
        }
    }
}
