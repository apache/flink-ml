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

package org.apache.flink.ml.classification.linearsvc;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.lossfunc.HingeLoss;
import org.apache.flink.ml.common.optimizer.Optimizer;
import org.apache.flink.ml.common.optimizer.SGD;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * An Estimator which implements the linear support vector classification.
 *
 * <p>See https://en.wikipedia.org/wiki/Support-vector_machine#Linear_SVM.
 */
public class LinearSVC implements Estimator<LinearSVC, LinearSVCModel>, LinearSVCParams<LinearSVC> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public LinearSVC() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings({"rawTypes", "ConstantConditions"})
    public LinearSVCModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<LabeledPointWithWeight> trainData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                dataPoint -> {
                                    double weight =
                                            getWeightCol() == null
                                                    ? 1.0
                                                    : ((Number) dataPoint.getField(getWeightCol()))
                                                            .doubleValue();
                                    double label =
                                            ((Number) dataPoint.getField(getLabelCol()))
                                                    .doubleValue();
                                    Preconditions.checkState(
                                            Double.compare(0.0, label) == 0
                                                    || Double.compare(1.0, label) == 0,
                                            "LinearSVC only supports binary classification. But detected label: %s.",
                                            label);
                                    DenseIntDoubleVector features =
                                            ((IntDoubleVector) dataPoint.getField(getFeaturesCol()))
                                                    .toDense();
                                    return new LabeledPointWithWeight(features, label, weight);
                                });

        DataStream<DenseIntDoubleVector> initModelData =
                DataStreamUtils.reduce(
                                trainData.map(x -> x.getFeatures().size()),
                                (ReduceFunction<Integer>)
                                        (t0, t1) -> {
                                            Preconditions.checkState(
                                                    t0.equals(t1),
                                                    "The training data should all have same dimensions.");
                                            return t0;
                                        })
                        .map(DenseIntDoubleVector::new);

        Optimizer optimizer =
                new SGD(
                        getMaxIter(),
                        getLearningRate(),
                        getGlobalBatchSize(),
                        getTol(),
                        getReg(),
                        getElasticNet());
        DataStream<DenseIntDoubleVector> rawModelData =
                optimizer.optimize(initModelData, trainData, HingeLoss.INSTANCE);

        DataStream<LinearSVCModelData> modelData = rawModelData.map(LinearSVCModelData::new);
        LinearSVCModel model = new LinearSVCModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static LinearSVC load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
