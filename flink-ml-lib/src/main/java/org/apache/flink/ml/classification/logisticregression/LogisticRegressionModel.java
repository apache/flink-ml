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

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** A Model which classifies data using the model data computed by {@link LogisticRegression}. */
public class LogisticRegressionModel
        implements Model<LogisticRegressionModel>,
                LogisticRegressionModelParams<LogisticRegressionModel> {

    private Map<Param<?>, Object> paramMap = new HashMap<>();

    private Table modelDataTable;

    public LogisticRegressionModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                LogisticRegressionModelData.getModelDataStream(modelDataTable),
                path,
                new LogisticRegressionModelData.ModelDataEncoder());
    }

    public static LogisticRegressionModel load(StreamExecutionEnvironment env, String path)
            throws IOException {
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        LogisticRegressionModel model = ReadWriteUtils.loadStageParam(path);
        DataStream<LogisticRegressionModelData> modelData =
                ReadWriteUtils.loadModelData(
                        env, path, new LogisticRegressionModelData.ModelDataDecoder());
        return model.setModelData(tEnv.fromDataStream(modelData));
    }

    @Override
    public LogisticRegressionModel setModelData(Table... inputs) {
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> inputStream = tEnv.toDataStream(inputs[0]);
        final String broadcastModelKey = "broadcastModelKey";
        DataStream<LogisticRegressionModelData> modelDataStream =
                LogisticRegressionModelData.getModelDataStream(modelDataTable);
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                BasicTypeInfo.DOUBLE_TYPE_INFO,
                                TypeInformation.of(DenseVector.class)),
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldNames(),
                                getPredictionCol(),
                                getRawPredictionCol()));
        DataStream<Row> predictionResult =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputStream),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            DataStream inputData = inputList.get(0);
                            return inputData.map(
                                    new PredictLabelFunction(broadcastModelKey, getFeaturesCol()),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    /** A utility function used for prediction. */
    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {

        private final String broadcastModelKey;

        private final String featuresCol;

        private DenseVector coefficient;

        public PredictLabelFunction(String broadcastModelKey, String featuresCol) {
            this.broadcastModelKey = broadcastModelKey;
            this.featuresCol = featuresCol;
        }

        @Override
        public Row map(Row dataPoint) {
            if (coefficient == null) {
                LogisticRegressionModelData modelData =
                        (LogisticRegressionModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                coefficient = modelData.coefficient;
            }
            DenseVector features = (DenseVector) dataPoint.getField(featuresCol);
            Tuple2<Double, DenseVector> predictionResult = predictRaw(features, coefficient);
            return Row.join(dataPoint, Row.of(predictionResult.f0), Row.of(predictionResult.f1));
        }
    }

    /**
     * The main logic that predicts one input record.
     *
     * @param feature The input feature.
     * @param coefficient The model parameters.
     * @return The prediction label and the raw probabilities.
     */
    private static Tuple2<Double, DenseVector> predictRaw(
            DenseVector feature, DenseVector coefficient) {
        double dotValue = BLAS.dot(feature, coefficient);
        double prob = 1 - 1.0 / (1.0 + Math.exp(dotValue));
        return new Tuple2<>(dotValue >= 0 ? 1. : 0., Vectors.dense(1 - prob, prob));
    }
}
