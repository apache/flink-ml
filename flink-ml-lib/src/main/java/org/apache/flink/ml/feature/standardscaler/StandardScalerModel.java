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

package org.apache.flink.ml.feature.standardscaler;

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

/** A Model which transforms data using the model data computed by {@link StandardScaler}. */
public class StandardScalerModel
        implements Model<StandardScalerModel>, StandardScalerParams<StandardScalerModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public StandardScalerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings("unchecked, rawtypes")
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
        DataStream<StandardScalerModelData> modelDataStream =
                StandardScalerModelData.getModelDataStream(modelDataTable);

        DataStream<Row> predictionResult =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputStream),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            DataStream inputData = inputList.get(0);
                            return inputData.map(
                                    new PredictOutputFunction(
                                            broadcastModelKey,
                                            getInputCol(),
                                            getWithMean(),
                                            getWithStd()),
                                    outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    /** A utility function used for prediction. */
    private static class PredictOutputFunction extends RichMapFunction<Row, Row> {
        private final String broadcastModelKey;
        private final String inputCol;
        private final boolean withMean;
        private final boolean withStd;
        private DenseIntDoubleVector mean;
        private DenseIntDoubleVector scale;

        public PredictOutputFunction(
                String broadcastModelKey, String inputCol, boolean withMean, boolean withStd) {
            this.broadcastModelKey = broadcastModelKey;
            this.inputCol = inputCol;
            this.withMean = withMean;
            this.withStd = withStd;
        }

        @Override
        public Row map(Row dataPoint) {
            if (mean == null) {
                StandardScalerModelData modelData =
                        (StandardScalerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                mean = modelData.mean;
                DenseIntDoubleVector std = modelData.std;

                if (withStd) {
                    scale = std;
                    double[] scaleValues = scale.values;
                    for (int i = 0; i < scaleValues.length; i++) {
                        scaleValues[i] = scaleValues[i] == 0 ? 0 : 1 / scaleValues[i];
                    }
                }
            }

            IntDoubleVector outputVec = ((IntDoubleVector) (dataPoint.getField(inputCol))).clone();
            if (withMean) {
                outputVec = outputVec.toDense();
                BLAS.axpy(-1, mean, (DenseIntDoubleVector) outputVec);
            }
            if (withStd) {
                BLAS.hDot(scale, outputVec);
            }

            return Row.join(dataPoint, Row.of(outputVec));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                StandardScalerModelData.getModelDataStream(modelDataTable),
                path,
                new StandardScalerModelData.ModelDataEncoder());
    }

    public static StandardScalerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        StandardScalerModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new StandardScalerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public StandardScalerModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }
}
