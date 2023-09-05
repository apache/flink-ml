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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A Model which predicts data using the model data computed by {@link Als}. */
public class AlsModel implements Model<AlsModel>, AlsModelParams<AlsModel> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    protected Table modelDataTable;

    public AlsModel() {
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
        DataStream<AlsModelData> modelDataStream = AlsModelData.getModelDataStream(modelDataTable);
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
                                    new PredictLabelFunction(
                                            broadcastModelKey, getUserCol(), getItemCol()),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    @Override
    public AlsModel setModelData(Table... inputs) {
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
                AlsModelData.getModelDataStream(modelDataTable),
                path,
                new AlsModelData.ModelDataEncoder());
    }

    public static AlsModel load(StreamTableEnvironment tEnv, String path) throws IOException {
        AlsModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(tEnv, path, new AlsModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /** A utility function used for prediction. */
    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {

        private final String broadcastModelKey;

        private final String userCol;
        private final String itemCol;

        public Map<Long, DenseVector> userFactors;
        public Map<Long, DenseVector> itemFactors;

        public PredictLabelFunction(String broadcastModelKey, String userCol, String itemCol) {
            this.broadcastModelKey = broadcastModelKey;
            this.userCol = userCol;
            this.itemCol = itemCol;
        }

        @Override
        public Row map(Row dataPoint) {
            if (userFactors == null) {
                List<AlsModelData> modelData =
                        getRuntimeContext().getBroadcastVariable(broadcastModelKey);

                List<Tuple2<Long, float[]>> uFactors = new ArrayList<>();
                List<Tuple2<Long, float[]>> iFactors = new ArrayList<>();
                for (AlsModelData data : modelData) {
                    uFactors.addAll(data.userFactors);
                    iFactors.addAll(data.itemFactors);
                }
                this.userFactors = new HashMap<>(uFactors.size());
                this.itemFactors = new HashMap<>(iFactors.size());
                for (Tuple2<Long, float[]> t2 : uFactors) {
                    double[] values = new double[t2.f1.length];
                    for (int i = 0; i < values.length; ++i) {
                        values[i] = t2.f1[i];
                    }
                    this.userFactors.put(t2.f0, new DenseVector(values));
                }
                for (Tuple2<Long, float[]> t2 : iFactors) {
                    double[] values = new double[t2.f1.length];
                    for (int i = 0; i < values.length; ++i) {
                        values[i] = t2.f1[i];
                    }
                    this.itemFactors.put(t2.f0, new DenseVector(values));
                }
            }

            Row predictionResult =
                    predictRating(dataPoint.getFieldAs(userCol), dataPoint.getFieldAs(itemCol));
            return Row.join(dataPoint, predictionResult);
        }

        private Row predictRating(long userId, long itemId) {
            DenseVector userFeat = userFactors.get(userId);
            DenseVector itemFeat = itemFactors.get(itemId);
            if (userFeat != null && itemFeat != null) {
                return Row.of(BLAS.dot(userFeat, itemFeat));
            } else {
                return Row.of(Double.NaN);
            }
        }
    }
}
