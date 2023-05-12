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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.feature.LabeledLargePointWithWeight;
import org.apache.flink.ml.common.lossfunc.BinaryLogisticLoss;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.common.ps.training.IterationStageList;
import org.apache.flink.ml.common.ps.training.ProcessStage;
import org.apache.flink.ml.common.ps.training.PullStage;
import org.apache.flink.ml.common.ps.training.PushStage;
import org.apache.flink.ml.common.ps.training.SerializableConsumer;
import org.apache.flink.ml.common.ps.training.TrainingContext;
import org.apache.flink.ml.common.ps.training.TrainingUtils;
import org.apache.flink.ml.common.updater.FTRL;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.util.ResettableIterator;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableFunction;
import org.apache.flink.util.function.SerializableSupplier;

import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the large scale logistic regression algorithm using FTRL optimizer.
 *
 * <p>See https://en.wikipedia.org/wiki/Logistic_regression.
 */
public class LogisticRegressionWithFtrl
        implements Estimator<LogisticRegressionWithFtrl, LogisticRegressionModel>,
                LogisticRegressionWithFtrlParams<LogisticRegressionWithFtrl> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public LogisticRegressionWithFtrl() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public LogisticRegressionModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String classificationType = getMultiClass();
        Preconditions.checkArgument(
                "auto".equals(classificationType) || "binomial".equals(classificationType),
                "Multinomial classification is not supported yet. Supported options: [auto, binomial].");
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<LabeledLargePointWithWeight> trainData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, LabeledLargePointWithWeight>)
                                        dataPoint -> {
                                            double weight =
                                                    getWeightCol() == null
                                                            ? 1.0
                                                            : ((Number)
                                                                            dataPoint.getField(
                                                                                    getWeightCol()))
                                                                    .doubleValue();
                                            double label =
                                                    ((Number) dataPoint.getField(getLabelCol()))
                                                            .doubleValue();
                                            boolean isBinomial =
                                                    Double.compare(0., label) == 0
                                                            || Double.compare(1., label) == 0;
                                            if (!isBinomial) {
                                                throw new RuntimeException(
                                                        "Multinomial classification is not supported yet. Supported options: [auto, binomial].");
                                            }
                                            Tuple2<long[], double[]> features =
                                                    dataPoint.getFieldAs(getFeaturesCol());
                                            return new LabeledLargePointWithWeight(
                                                    features, label, weight);
                                        });

        DataStream<Long> modelDim;
        if (getModelDim() > 0) {
            modelDim = trainData.getExecutionEnvironment().fromElements(getModelDim());
        } else {
            modelDim =
                    DataStreamUtils.reduce(
                                    trainData.map(x -> x.features.f0[x.features.f0.length - 1]),
                                    (ReduceFunction<Long>) Math::max)
                            .map((MapFunction<Long, Long>) value -> value + 1);
        }

        LogisticRegressionWithFtrlTrainingContext trainingContext =
                new LogisticRegressionWithFtrlTrainingContext(getParamMap());

        IterationStageList<LogisticRegressionWithFtrlTrainingContext> iterationStages =
                new IterationStageList<>(trainingContext);
        iterationStages
                .addTrainingStage(new ComputeIndices())
                .addTrainingStage(
                        new PullStage(
                                (SerializableSupplier<long[]>) () -> trainingContext.pullIndices,
                                (SerializableConsumer<double[]>)
                                        x -> trainingContext.pulledValues = x))
                .addTrainingStage(new ComputeGradients(BinaryLogisticLoss.INSTANCE))
                .addTrainingStage(
                        new PushStage(
                                (SerializableSupplier<long[]>) () -> trainingContext.pushIndices,
                                (SerializableSupplier<double[]>) () -> trainingContext.pushValues))
                .setTerminationCriteria(
                        (SerializableFunction<LogisticRegressionWithFtrlTrainingContext, Boolean>)
                                o -> o.iterationId >= getMaxIter());
        FTRL ftrl = new FTRL(getAlpha(), getBeta(), getReg(), getElasticNet());

        DataStream<Tuple3<Long, Long, double[]>> rawModelData =
                TrainingUtils.train(
                        modelDim,
                        trainData,
                        ftrl,
                        iterationStages,
                        getNumServers(),
                        getNumServerCores());

        final long modelVersion = 0L;

        DataStream<LogisticRegressionModelData> modelData =
                rawModelData.map(
                        tuple3 ->
                                new LogisticRegressionModelData(
                                        Vectors.dense(tuple3.f2),
                                        tuple3.f0,
                                        tuple3.f1,
                                        modelVersion));

        LogisticRegressionModel model =
                new LogisticRegressionModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static LogisticRegressionWithFtrl load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}

/**
 * An iteration stage that samples a batch of training data and computes the indices needed to
 * compute gradients.
 */
class ComputeIndices extends ProcessStage<LogisticRegressionWithFtrlTrainingContext> {

    @Override
    public void process(LogisticRegressionWithFtrlTrainingContext context) throws Exception {
        context.readInNextBatchData();
        context.pullIndices = computeIndices(context.batchData);
    }

    public static long[] computeIndices(List<LabeledLargePointWithWeight> dataPoints) {
        LongOpenHashSet indices = new LongOpenHashSet();
        for (LabeledLargePointWithWeight dataPoint : dataPoints) {
            long[] notZeros = dataPoint.features.f0;
            for (long index : notZeros) {
                indices.add(index);
            }
        }

        long[] sortedIndices = new long[indices.size()];
        Iterator<Long> iterator = indices.iterator();
        int i = 0;
        while (iterator.hasNext()) {
            sortedIndices[i++] = iterator.next();
        }
        Arrays.sort(sortedIndices);
        return sortedIndices;
    }
}

/**
 * An iteration stage that uses the pulled model values and sampled batch data to compute the
 * gradients.
 */
class ComputeGradients extends ProcessStage<LogisticRegressionWithFtrlTrainingContext> {
    private final LossFunc lossFunc;

    public ComputeGradients(LossFunc lossFunc) {
        this.lossFunc = lossFunc;
    }

    @Override
    public void process(LogisticRegressionWithFtrlTrainingContext context) throws IOException {
        long[] indices = ComputeIndices.computeIndices(context.batchData);
        double[] pulledModelValues = context.pulledValues;
        double[] gradients = computeGradient(context.batchData, indices, pulledModelValues);

        context.pushIndices = indices;
        context.pushValues = gradients;
    }

    private double[] computeGradient(
            List<LabeledLargePointWithWeight> batchData,
            long[] sortedBatchIndices,
            double[] pulledModelValues) {
        Long2DoubleOpenHashMap coefficient = new Long2DoubleOpenHashMap(sortedBatchIndices.length);
        for (int i = 0; i < sortedBatchIndices.length; i++) {
            coefficient.put(sortedBatchIndices[i], pulledModelValues[i]);
        }
        Long2DoubleOpenHashMap cumGradients = new Long2DoubleOpenHashMap(sortedBatchIndices.length);

        for (LabeledLargePointWithWeight dataPoint : batchData) {
            double dot = dot(dataPoint.features, coefficient);
            double multiplier = lossFunc.computeGradient(dataPoint.label, dot) * dataPoint.weight;

            long[] featureIndices = dataPoint.features.f0;
            double[] featureValues = dataPoint.features.f1;
            double z;
            for (int i = 0; i < featureIndices.length; i++) {
                long currentIndex = featureIndices[i];
                z = featureValues[i] * multiplier + cumGradients.getOrDefault(currentIndex, 0.);
                cumGradients.put(currentIndex, z);
            }
        }
        double[] cumGradientValues = new double[sortedBatchIndices.length];
        for (int i = 0; i < sortedBatchIndices.length; i++) {
            cumGradientValues[i] = cumGradients.get(sortedBatchIndices[i]);
        }
        return cumGradientValues;
    }

    private static double dot(
            Tuple2<long[], double[]> features, Long2DoubleOpenHashMap coefficient) {
        double dot = 0;
        for (int i = 0; i < features.f0.length; i++) {
            dot += features.f1[i] * coefficient.get(features.f0[i]);
        }
        return dot;
    }
}

/** The context information of local computing process. */
class LogisticRegressionWithFtrlTrainingContext
        implements TrainingContext,
                LogisticRegressionWithFtrlParams<LogisticRegressionWithFtrlTrainingContext> {
    /** Parameters of LogisticRegressionWithFtrl. */
    private final Map<Param<?>, Object> paramMap;
    /** Current iteration id. */
    int iterationId;
    /** The local batch size. */
    private int localBatchSize = -1;
    /** The training data. */
    private ResettableIterator<LabeledLargePointWithWeight> trainData;
    /** The batch of training data for computing gradients. */
    List<LabeledLargePointWithWeight> batchData;

    private ListState<LabeledLargePointWithWeight> batchDataState;

    /** The placeholder for indices to pull for each iteration. */
    long[] pullIndices;
    /** The placeholder for the pulled values for each iteration. */
    double[] pulledValues;
    /** The placeholder for indices to push for each iteration. */
    long[] pushIndices;
    /** The placeholder for values to push for each iteration. */
    double[] pushValues;

    public LogisticRegressionWithFtrlTrainingContext(Map<Param<?>, Object> paramMap) {
        this.paramMap = paramMap;
    }

    @Override
    public void setIterationId(int iterationId) {
        this.iterationId = iterationId;
    }

    @Override
    public void setWorldInfo(int workerId, int numWorkers) {
        int globalBatchSize = getGlobalBatchSize();
        this.localBatchSize = globalBatchSize / numWorkers;
        if (globalBatchSize % numWorkers > workerId) {
            localBatchSize++;
        }
        this.batchData = new ArrayList<>(localBatchSize);
    }

    @Override
    public void setTrainData(ResettableIterator<?> trainData) {
        this.trainData = (ResettableIterator<LabeledLargePointWithWeight>) trainData;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        batchDataState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "batchDataState",
                                        TypeInformation.of(LabeledLargePointWithWeight.class)));

        Iterator<LabeledLargePointWithWeight> batchDataIterator = batchDataState.get().iterator();
        if (batchDataIterator.hasNext()) {
            batchData = IteratorUtils.toList(batchDataIterator);
        }
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        batchDataState.clear();
        if (batchData.size() > 0) {
            batchDataState.addAll(batchData);
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /** Reads in next batch of training data. */
    public void readInNextBatchData() throws IOException {
        batchData.clear();
        int i = 0;
        while (i < localBatchSize && trainData.hasNext()) {
            batchData.add(trainData.next());
            i++;
        }
        if (!trainData.hasNext()) {
            trainData.reset();
        }
    }
}
