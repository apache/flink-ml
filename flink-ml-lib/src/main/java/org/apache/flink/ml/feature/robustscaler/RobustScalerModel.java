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

package org.apache.flink.ml.feature.robustscaler;

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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** A Model which transforms data using the model data computed by {@link RobustScaler}. */
public class RobustScalerModel
        implements Model<RobustScalerModel>, RobustScalerModelParams<RobustScalerModel> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public RobustScalerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> inputStream = tEnv.toDataStream(inputs[0]);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));
        final String broadcastModelKey = "broadcastModelKey";
        DataStream<RobustScalerModelData> modelDataStream =
                RobustScalerModelData.getModelDataStream(modelDataTable);

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputStream),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            DataStream inputData = inputList.get(0);
                            return inputData.map(
                                    new PredictOutputFunction(
                                            broadcastModelKey,
                                            getInputCol(),
                                            getWithCentering(),
                                            getWithScaling()),
                                    outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(output)};
    }

    /** This operator loads model data and predicts result. */
    private static class PredictOutputFunction extends RichMapFunction<Row, Row> {
        private final String broadcastModelKey;
        private final String inputCol;
        private final boolean withCentering;
        private final boolean withScaling;

        private DenseIntDoubleVector medians;
        private DenseIntDoubleVector scales;

        public PredictOutputFunction(
                String broadcastModelKey,
                String inputCol,
                boolean withCentering,
                boolean withScaling) {
            this.broadcastModelKey = broadcastModelKey;
            this.inputCol = inputCol;
            this.withCentering = withCentering;
            this.withScaling = withScaling;
        }

        @Override
        public Row map(Row row) throws Exception {
            if (medians == null) {
                RobustScalerModelData modelData =
                        (RobustScalerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                medians = modelData.medians;
                scales =
                        new DenseIntDoubleVector(
                                Arrays.stream(modelData.ranges.values)
                                        .map(range -> range == 0 ? 0 : 1 / range)
                                        .toArray());
            }
            DenseIntDoubleVector outputVec =
                    ((IntDoubleVector) row.getField(inputCol)).clone().toDense();
            Preconditions.checkState(
                    medians.size() == outputVec.size(),
                    "Number of features must be %s but got %s.",
                    medians.size(),
                    outputVec.size());

            if (withCentering) {
                BLAS.axpy(-1, medians, outputVec);
            }
            if (withScaling) {
                BLAS.hDot(scales, outputVec);
            }
            return Row.join(row, Row.of(outputVec));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                RobustScalerModelData.getModelDataStream(modelDataTable),
                path,
                new RobustScalerModelData.ModelDataEncoder());
    }

    public static RobustScalerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        RobustScalerModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new RobustScalerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public RobustScalerModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        this.modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
