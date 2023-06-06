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

package org.apache.flink.ml.regression.linearregression;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
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

/** A Model which predicts data using the model data computed by {@link LinearRegression}. */
public class LinearRegressionModel
        implements Model<LinearRegressionModel>,
                LinearRegressionModelParams<LinearRegressionModel> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    private Table modelDataTable;

    public LinearRegressionModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> inputStream = tEnv.toDataStream(inputs[0]);
        final String broadcastModelKey = "broadcastModelKey";
        DataStream<LinearRegressionModelData> modelDataStream =
                LinearRegressionModelData.getModelDataStream(modelDataTable);
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(), BasicTypeInfo.DOUBLE_TYPE_INFO),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol()));
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

    @Override
    public LinearRegressionModel setModelData(Table... inputs) {
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                LinearRegressionModelData.getModelDataStream(modelDataTable),
                path,
                new LinearRegressionModelData.ModelDataEncoder());
    }

    public static LinearRegressionModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        LinearRegressionModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new LinearRegressionModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /** A utility function used for prediction. */
    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {

        private final String broadcastModelKey;

        private final String featuresCol;

        private DenseIntDoubleVector coefficient;

        public PredictLabelFunction(String broadcastModelKey, String featuresCol) {
            this.broadcastModelKey = broadcastModelKey;
            this.featuresCol = featuresCol;
        }

        @Override
        public Row map(Row dataPoint) {
            if (coefficient == null) {
                LinearRegressionModelData modelData =
                        (LinearRegressionModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                coefficient = modelData.coefficient;
            }
            DenseIntDoubleVector features =
                    ((IntDoubleVector) dataPoint.getField(featuresCol)).toDense();
            Row predictionResult = predictOneDataPoint(features, coefficient);
            return Row.join(dataPoint, predictionResult);
        }
    }

    /**
     * The main logic that predicts one input data point.
     *
     * @param feature The input feature.
     * @param coefficient The model parameters.
     * @return The prediction label and the raw probabilities.
     */
    private static Row predictOneDataPoint(
            DenseIntDoubleVector feature, DenseIntDoubleVector coefficient) {
        return Row.of(BLAS.dot(feature, coefficient));
    }
}
