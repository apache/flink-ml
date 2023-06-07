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
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.lossfunc.BinaryLogisticLoss;
import org.apache.flink.ml.common.ps.training.ComputeGradients;
import org.apache.flink.ml.common.ps.training.ComputeIndices;
import org.apache.flink.ml.common.ps.training.IterationStageList;
import org.apache.flink.ml.common.ps.training.MiniBatchMLSession;
import org.apache.flink.ml.common.ps.training.PullStage;
import org.apache.flink.ml.common.ps.training.PushStage;
import org.apache.flink.ml.common.ps.training.SerializableConsumer;
import org.apache.flink.ml.common.ps.training.TrainingUtils;
import org.apache.flink.ml.common.ps.updater.FTRL;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.LongDoubleVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableFunction;
import org.apache.flink.util.function.SerializableSupplier;

import java.io.IOException;
import java.util.HashMap;
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

        DataStream<LabeledPointWithWeight> trainData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, LabeledPointWithWeight>)
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
                                            Vector features =
                                                    dataPoint.getFieldAs(getFeaturesCol());
                                            return new LabeledPointWithWeight(
                                                    features, label, weight);
                                        });

        DataStream<Long> modelDim;
        if (getModelDim() > 0) {
            modelDim = trainData.getExecutionEnvironment().fromElements(getModelDim());
        } else {
            modelDim =
                    DataStreamUtils.reduce(
                                    trainData.map(
                                            x -> {
                                                Vector feature = x.features;
                                                long dim;
                                                if (feature instanceof IntDoubleVector) {
                                                    dim =
                                                            ((IntDoubleVector) feature)
                                                                    .size()
                                                                    .intValue();
                                                } else {
                                                    dim =
                                                            ((LongDoubleVector) feature)
                                                                    .size()
                                                                    .longValue();
                                                }
                                                return dim;
                                            }),
                                    (ReduceFunction<Long>) Math::max)
                            .map((MapFunction<Long, Long>) value -> value);
        }

        MiniBatchMLSession<LabeledPointWithWeight> mlSession =
                new MiniBatchMLSession<>(
                        getGlobalBatchSize(), TypeInformation.of(LabeledPointWithWeight.class));

        IterationStageList<MiniBatchMLSession<LabeledPointWithWeight>> iterationStages =
                new IterationStageList<>(mlSession);
        iterationStages
                .addStage(new ComputeIndices())
                .addStage(
                        new PullStage(
                                (SerializableSupplier<long[]>) () -> mlSession.pullIndices,
                                (SerializableConsumer<double[]>) x -> mlSession.pulledValues = x))
                .addStage(new ComputeGradients(BinaryLogisticLoss.INSTANCE))
                .addStage(
                        new PushStage(
                                (SerializableSupplier<long[]>) () -> mlSession.pushIndices,
                                (SerializableSupplier<double[]>) () -> mlSession.pushValues))
                .setTerminationCriteria(
                        (SerializableFunction<MiniBatchMLSession<LabeledPointWithWeight>, Boolean>)
                                o -> o.iterationId >= getMaxIter());
        FTRL ftrl = new FTRL(getAlpha(), getBeta(), getReg(), getElasticNet());

        DataStream<Tuple3<Long, Long, double[]>> rawModelData =
                TrainingUtils.train(modelDim, trainData, ftrl, iterationStages, getNumServers());

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
