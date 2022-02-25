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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIterOrTol;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * An Estimator which implements the logistic regression algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/Logistic_regression.
 */
public class LogisticRegression
        implements Estimator<LogisticRegression, LogisticRegressionModel>,
                LogisticRegressionParams<LogisticRegression> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public LogisticRegression() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static LogisticRegression load(StreamExecutionEnvironment env, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    @SuppressWarnings({"rawTypes", "ConstantConditions"})
    public LogisticRegressionModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String classificationType = getMultiClass();
        Preconditions.checkArgument(
                "auto".equals(classificationType) || "binomial".equals(classificationType),
                "Multinomial classification is not supported yet. Supported options: [auto, binomial].");
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<LabeledPointWithWeight> trainData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                dataPoint -> {
                                    Double weight =
                                            getWeightCol() == null
                                                    ? 1.0
                                                    : (Double) dataPoint.getField(getWeightCol());
                                    Double label = (Double) dataPoint.getField(getLabelCol());
                                    boolean isBinomial =
                                            Double.compare(0., label) == 0
                                                    || Double.compare(1., label) == 0;
                                    if (!isBinomial) {
                                        throw new RuntimeException(
                                                "Multinomial classification is not supported yet. Supported options: [auto, binomial].");
                                    }
                                    DenseVector features =
                                            (DenseVector) dataPoint.getField(getFeaturesCol());
                                    return new LabeledPointWithWeight(features, label, weight);
                                });
        DataStream<double[]> initModelData =
                trainData
                        .transform("getModelDim", BasicTypeInfo.INT_TYPE_INFO, new GetModelDim())
                        .setParallelism(1)
                        .broadcast()
                        .map(double[]::new);

        DataStream<LogisticRegressionModelData> modelData = train(trainData, initModelData);
        LogisticRegressionModel model =
                new LogisticRegressionModel().setModelData(tEnv.fromDataStream(modelData));
        ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /** Gets the dimension of the model data. */
    private static class GetModelDim extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<LabeledPointWithWeight, Integer>, BoundedOneInput {

        private int dim = 0;

        private ListState<Integer> dimState;

        @Override
        public void endInput() {
            output.collect(new StreamRecord<>(dim));
        }

        @Override
        public void processElement(StreamRecord<LabeledPointWithWeight> streamRecord) {
            if (dim == 0) {
                dim = streamRecord.getValue().getFeatures().size();
            } else {
                if (dim != streamRecord.getValue().getFeatures().size()) {
                    throw new RuntimeException(
                            "The training data should all have same dimensions.");
                }
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            dimState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "dimState", BasicTypeInfo.INT_TYPE_INFO));
            dim = OperatorStateUtils.getUniqueElement(dimState, "dimState").orElse(0);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            dimState.clear();
            dimState.add(dim);
        }
    }

    /**
     * Does machine learning training on the input data with the initialized model data.
     *
     * @param trainData The training data.
     * @param initModelData The initialized model.
     * @return The trained model data.
     */
    private DataStream<LogisticRegressionModelData> train(
            DataStream<LabeledPointWithWeight> trainData, DataStream<double[]> initModelData) {
        LogisticGradient logisticGradient = new LogisticGradient(getReg());
        DataStreamList resultList =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(initModelData),
                        ReplayableDataStreamList.notReplay(trainData),
                        IterationConfig.newBuilder().build(),
                        new TrainIterationBody(
                                logisticGradient,
                                getGlobalBatchSize(),
                                getLearningRate(),
                                getMaxIter(),
                                getTol()));
        return resultList.get(0);
    }

    /** The iteration implementation for training process. */
    private static class TrainIterationBody implements IterationBody {

        private final LogisticGradient logisticGradient;

        private final int globalBatchSize;

        private final double learningRate;

        private final int maxIter;

        private final double tol;

        public TrainIterationBody(
                LogisticGradient logisticGradient,
                int globalBatchSize,
                double learningRate,
                int maxIter,
                double tol) {
            this.logisticGradient = logisticGradient;
            this.globalBatchSize = globalBatchSize;
            this.learningRate = learningRate;
            this.maxIter = maxIter;
            this.tol = tol;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            // The variable stream at the first iteration is the initialized model data.
            // In the following iterations, it contains: the computed gradient, weightSum and
            // lossSum.
            DataStream<double[]> variableStream = variableStreams.get(0);
            DataStream<LabeledPointWithWeight> trainData = dataStreams.get(0);
            final OutputTag<LogisticRegressionModelData> modelDataOutputTag =
                    new OutputTag<LogisticRegressionModelData>("MODEL_OUTPUT") {};
            SingleOutputStreamOperator<double[]> gradientAndWeightAndLoss =
                    trainData
                            .connect(variableStream)
                            .transform(
                                    "CacheDataAndDoTrain",
                                    PrimitiveArrayTypeInfo.DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO,
                                    new CacheDataAndDoTrain(
                                            logisticGradient,
                                            globalBatchSize,
                                            learningRate,
                                            modelDataOutputTag));
            DataStreamList feedbackVariableStream =
                    IterationBody.forEachRound(
                            DataStreamList.of(gradientAndWeightAndLoss),
                            input -> {
                                DataStream<double[]> feedback =
                                        DataStreamUtils.allReduceSum(input.get(0));
                                return DataStreamList.of(feedback);
                            });
            DataStream<Integer> terminationCriteria =
                    feedbackVariableStream
                            .get(0)
                            .map(
                                    reducedGradientAndWeightAndLoss -> {
                                        double[] value = (double[]) reducedGradientAndWeightAndLoss;
                                        return value[value.length - 1] / value[value.length - 2];
                                    })
                            .flatMap(new TerminateOnMaxIterOrTol(maxIter, tol));
            return new IterationBodyResult(
                    DataStreamList.of(feedbackVariableStream.get(0)),
                    DataStreamList.of(gradientAndWeightAndLoss.getSideOutput(modelDataOutputTag)),
                    terminationCriteria);
        }
    }

    /**
     * A stream operator that caches the training data in the first iteration and updates the model
     * using gradients iteratively. The first input is the training data, and the second input is
     * the initialized model data or feedback of gradient, weight and loss.
     */
    private static class CacheDataAndDoTrain extends AbstractStreamOperator<double[]>
            implements TwoInputStreamOperator<LabeledPointWithWeight, double[], double[]>,
                    IterationListener<double[]> {

        private final int globalBatchSize;

        private int localBatchSize;

        private final double learningRate;

        private final LogisticGradient logisticGradient;

        private DenseVector gradient;

        private DenseVector coefficient;

        private int coefficientDim;

        private ListState<DenseVector> coefficientState;

        private List<LabeledPointWithWeight> trainData;

        private ListState<LabeledPointWithWeight> trainDataState;

        private final Random random = new Random(2021);

        private List<LabeledPointWithWeight> miniBatchData;

        /** The buffer for feedback record: {gradient, weightSum, loss}. */
        private double[] feedbackBuffer;

        private ListState<double[]> feedbackBufferState;

        private final OutputTag<LogisticRegressionModelData> modelDataOutputTag;

        public CacheDataAndDoTrain(
                LogisticGradient logisticGradient,
                int globalBatchSize,
                double learningRate,
                OutputTag<LogisticRegressionModelData> modelDataOutputTag) {
            this.logisticGradient = logisticGradient;
            this.globalBatchSize = globalBatchSize;
            this.learningRate = learningRate;
            this.modelDataOutputTag = modelDataOutputTag;
        }

        @Override
        public void open() {
            int numTasks = getRuntimeContext().getNumberOfParallelSubtasks();
            int taskId = getRuntimeContext().getIndexOfThisSubtask();
            localBatchSize = globalBatchSize / numTasks;
            if (globalBatchSize % numTasks > taskId) {
                localBatchSize++;
            }
            this.miniBatchData = new ArrayList<>(localBatchSize);
        }

        private List<LabeledPointWithWeight> getMiniBatchData(
                List<LabeledPointWithWeight> fullBatchData, int batchSize) {
            miniBatchData.clear();
            for (int i = 0; i < batchSize; i++) {
                miniBatchData.add(fullBatchData.get(random.nextInt(fullBatchData.size())));
            }
            return miniBatchData;
        }

        private void updateModel() {
            System.arraycopy(feedbackBuffer, 0, gradient.values, 0, gradient.size());
            double weightSum = feedbackBuffer[coefficientDim];
            BLAS.axpy(-learningRate / weightSum, gradient, coefficient);
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<double[]> collector)
                throws Exception {
            if (epochWatermark == 0) {
                coefficient = new DenseVector(feedbackBuffer);
                coefficientDim = coefficient.size();
                feedbackBuffer = new double[coefficientDim + 2];
                gradient = new DenseVector(coefficientDim);
            } else {
                updateModel();
            }
            Arrays.fill(gradient.values, 0);
            if (trainData == null) {
                trainData = IteratorUtils.toList(trainDataState.get().iterator());
            }
            if (trainData.size() > 0) {
                miniBatchData = getMiniBatchData(trainData, localBatchSize);
                Tuple2<Double, Double> weightSumAndLossSum =
                        logisticGradient.computeLoss(miniBatchData, coefficient);
                logisticGradient.computeGradient(miniBatchData, coefficient, gradient);
                System.arraycopy(gradient.values, 0, feedbackBuffer, 0, gradient.size());
                feedbackBuffer[coefficientDim] = weightSumAndLossSum.f0;
                feedbackBuffer[coefficientDim + 1] = weightSumAndLossSum.f1;
                collector.collect(feedbackBuffer);
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<double[]> collector) {
            trainDataState.clear();
            coefficientState.clear();
            feedbackBufferState.clear();
            if (getRuntimeContext().getIndexOfThisSubtask() == 0) {
                updateModel();
                context.output(modelDataOutputTag, new LogisticRegressionModelData(coefficient));
            }
        }

        @Override
        public void processElement1(StreamRecord<LabeledPointWithWeight> streamRecord)
                throws Exception {
            trainDataState.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<double[]> streamRecord) {
            feedbackBuffer = streamRecord.getValue();
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            trainDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "trainDataState",
                                            TypeInformation.of(LabeledPointWithWeight.class)));
            coefficientState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "coefficientState",
                                            TypeInformation.of(DenseVector.class)));
            OperatorStateUtils.getUniqueElement(coefficientState, "coefficientState")
                    .ifPresent(x -> coefficient = x);
            feedbackBufferState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "feedbackBufferState",
                                            PrimitiveArrayTypeInfo
                                                    .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO));
            OperatorStateUtils.getUniqueElement(feedbackBufferState, "feedbackBufferState")
                    .ifPresent(x -> feedbackBuffer = x);
            if (coefficient != null) {
                coefficientDim = coefficient.size();
                gradient = new DenseVector(new double[coefficientDim]);
            }
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            coefficientState.clear();
            if (coefficient != null) {
                coefficientState.add(coefficient);
            }
            feedbackBufferState.clear();
            if (feedbackBuffer != null) {
                feedbackBufferState.add(feedbackBuffer);
            }
        }
    }
}
