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

package org.apache.flink.ml.feature.maxabsscaler;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
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

/** A Model which transforms data using the model data computed by {@link MaxAbsScaler}. */
public class MaxAbsScalerModel
        implements Model<MaxAbsScalerModel>, MaxAbsScalerParams<MaxAbsScalerModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public MaxAbsScalerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public MaxAbsScalerModel setModelData(Table... inputs) {
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

        DataStream<Row> data = tEnv.toDataStream(inputs[0]);
        DataStream<MaxAbsScalerModelData> maxAbsScalerModel =
                MaxAbsScalerModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(data),
                        Collections.singletonMap(broadcastModelKey, maxAbsScalerModel),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new PredictOutputFunction(broadcastModelKey, getInputCol()),
                                    outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                MaxAbsScalerModelData.getModelDataStream(modelDataTable),
                path,
                new MaxAbsScalerModelData.ModelDataEncoder());
    }

    /**
     * Loads model data from path.
     *
     * @param tEnv Stream table environment.
     * @param path Model path.
     * @return MaxAbsScalerModel model.
     */
    public static MaxAbsScalerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        MaxAbsScalerModel model = ReadWriteUtils.loadStageParam(path);

        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new MaxAbsScalerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    /** This function loads model data and predicts result. */
    private static class PredictOutputFunction extends RichMapFunction<Row, Row> {
        private final String inputCol;
        private final String broadcastKey;
        private DenseIntDoubleVector scaleVector;

        public PredictOutputFunction(String broadcastKey, String inputCol) {
            this.broadcastKey = broadcastKey;
            this.inputCol = inputCol;
        }

        @Override
        public Row map(Row row) {
            if (scaleVector == null) {
                MaxAbsScalerModelData maxAbsScalerModelData =
                        (MaxAbsScalerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                scaleVector = maxAbsScalerModelData.maxVector;

                for (int i = 0; i < scaleVector.size(); ++i) {
                    if (scaleVector.values[i] != 0) {
                        scaleVector.values[i] = 1.0 / scaleVector.values[i];
                    } else {
                        scaleVector.values[i] = 1.0;
                    }
                }
            }

            IntDoubleVector inputVec = row.getFieldAs(inputCol);
            IntDoubleVector outputVec = inputVec.clone();
            BLAS.hDot(scaleVector, outputVec);
            return Row.join(row, Row.of(outputVec));
        }
    }
}
