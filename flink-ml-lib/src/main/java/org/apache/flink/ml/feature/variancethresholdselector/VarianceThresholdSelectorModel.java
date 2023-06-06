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

package org.apache.flink.ml.feature.variancethresholdselector;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.util.VectorUtils;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
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

/**
 * A Model which removes low-variance data using the model data computed by {@link
 * VarianceThresholdSelector}.
 */
public class VarianceThresholdSelectorModel
        implements Model<VarianceThresholdSelectorModel>,
                VarianceThresholdSelectorModelParams<VarianceThresholdSelectorModel> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public VarianceThresholdSelectorModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public VarianceThresholdSelectorModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
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

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                VarianceThresholdSelectorModelData.getModelDataStream(modelDataTable),
                path,
                new VarianceThresholdSelectorModelData.ModelDataEncoder());
    }

    public static VarianceThresholdSelectorModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        VarianceThresholdSelectorModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new VarianceThresholdSelectorModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> data = tEnv.toDataStream(inputs[0]);
        DataStream<VarianceThresholdSelectorModelData> varianceThresholdSelectorModel =
                VarianceThresholdSelectorModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(data),
                        Collections.singletonMap(broadcastModelKey, varianceThresholdSelectorModel),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new PredictOutputFunction(getInputCol(), broadcastModelKey),
                                    outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(output)};
    }

    /** This operator loads model data and predicts result. */
    private static class PredictOutputFunction extends RichMapFunction<Row, Row> {

        private final String inputCol;
        private final String broadcastKey;
        private int expectedNumOfFeatures;
        private int[] indices = null;

        public PredictOutputFunction(String inputCol, String broadcastKey) {
            this.inputCol = inputCol;
            this.broadcastKey = broadcastKey;
        }

        @Override
        public Row map(Row row) {
            if (indices == null) {
                VarianceThresholdSelectorModelData varianceThresholdSelectorModelData =
                        (VarianceThresholdSelectorModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                expectedNumOfFeatures = varianceThresholdSelectorModelData.numOfFeatures;
                indices =
                        Arrays.stream(varianceThresholdSelectorModelData.indices)
                                .sorted()
                                .toArray();
            }

            IntDoubleVector inputVec = ((IntDoubleVector) row.getField(inputCol));
            Preconditions.checkArgument(
                    inputVec.size() == expectedNumOfFeatures,
                    "%s has %s features, but VarianceThresholdSelector is expecting %s features as input.",
                    inputCol,
                    inputVec.size(),
                    expectedNumOfFeatures);
            if (indices.length == 0) {
                return Row.join(row, Row.of(Vectors.dense()));
            } else {
                IntDoubleVector outputVec = VectorUtils.selectByIndices(inputVec, indices);
                return Row.join(row, Row.of(outputVec));
            }
        }
    }
}
