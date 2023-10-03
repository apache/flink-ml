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

package org.apache.flink.ml.anomalydetection.isolationforest;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.RowUtils;
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
import java.util.List;
import java.util.Map;

/**
 * A Model which detection anomaly data using the model data computed by {@link IsolationForest}.
 */
public class IsolationForestModel
        implements Model<IsolationForestModel>, IsolationForestModelParams<IsolationForestModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    private Table modelDataTable;

    public IsolationForestModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public IsolationForestModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<IsolationForestModelData> modelDataStream =
                IsolationForestModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(), BasicTypeInfo.INT_TYPE_INFO),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol()));

        DataStream<Row> predictionResult =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(tEnv.toDataStream(inputs[0])),
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
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                IsolationForestModelData.getModelDataStream(modelDataTable),
                path,
                new IsolationForestModelData.ModelDataEncoder());
    }

    public static IsolationForestModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new IsolationForestModelData.ModelDataDecoder());

        IsolationForestModel model = ReadWriteUtils.loadStageParam(path);
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
        private IsolationForestModelData modelData = null;
        public List<ITree> iTreeList;
        public Double center0;
        public Double center1;
        public int subSamplesSize;

        public PredictLabelFunction(String broadcastModelKey, String featuresCol) {
            this.broadcastModelKey = broadcastModelKey;
            this.featuresCol = featuresCol;
        }

        @Override
        public Row map(Row dataPoint) throws Exception {
            if (modelData == null) {
                modelData =
                        (IsolationForestModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                iTreeList = modelData.iForest.iTreeList;
                center0 = modelData.iForest.center0;
                center1 = modelData.iForest.center1;
                subSamplesSize = modelData.iForest.subSamplesSize;
            }

            DenseVector point = ((Vector) dataPoint.getField(featuresCol)).toDense();
            int predictId = predict(point);
            return RowUtils.append(dataPoint, predictId);
        }

        private int predict(DenseVector sampleData) throws Exception {
            double pathLengthSum = 0;
            int treesNumber = iTreeList.size();
            for (int j = 0; j < treesNumber; j++) {
                pathLengthSum += ITree.calculatePathLength(sampleData, iTreeList.get(j));
            }
            double pathLengthAvg = pathLengthSum / treesNumber;
            double cn = ITree.calculateCn(subSamplesSize);
            double score = Math.pow(2, -pathLengthAvg / cn);

            return Math.abs(score - center0) > Math.abs(score - center1) ? 1 : 0;
        }
    }
}
