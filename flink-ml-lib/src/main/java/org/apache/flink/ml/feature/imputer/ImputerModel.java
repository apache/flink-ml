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

package org.apache.flink.ml.feature.imputer;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
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

/** A Model which replaces the missing values using the model data computed by {@link Imputer}. */
public class ImputerModel implements Model<ImputerModel>, ImputerModelParams<ImputerModel> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public ImputerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public ImputerModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
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
                ImputerModelData.getModelDataStream(modelDataTable),
                path,
                new ImputerModelData.ModelDataEncoder());
    }

    public static ImputerModel load(StreamTableEnvironment tEnv, String path) throws IOException {
        ImputerModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(tEnv, path, new ImputerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String[] inputCols = getInputCols();
        String[] outputCols = getOutputCols();
        Preconditions.checkArgument(inputCols.length == outputCols.length);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> dataStream = tEnv.toDataStream(inputs[0]);
        DataStream<ImputerModelData> imputerModel =
                ImputerModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        TypeInformation<?>[] outputTypes = new TypeInformation[outputCols.length];
        Arrays.fill(outputTypes, BasicTypeInfo.DOUBLE_TYPE_INFO);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), outputTypes),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), outputCols));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(dataStream),
                        Collections.singletonMap(broadcastModelKey, imputerModel),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new PredictOutputFunction(
                                            getMissingValue(), inputCols, broadcastModelKey),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(output)};
    }

    /** This operator loads model data and predicts result. */
    private static class PredictOutputFunction extends RichMapFunction<Row, Row> {

        private final String[] inputCols;
        private final String broadcastKey;
        private final double missingValue;
        private Map<String, Double> surrogates;

        public PredictOutputFunction(double missingValue, String[] inputCols, String broadcastKey) {
            this.missingValue = missingValue;
            this.inputCols = inputCols;
            this.broadcastKey = broadcastKey;
        }

        @Override
        public Row map(Row row) throws Exception {
            if (surrogates == null) {
                ImputerModelData imputerModelData =
                        (ImputerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                surrogates = imputerModelData.surrogates;
                Arrays.stream(inputCols)
                        .forEach(
                                col ->
                                        Preconditions.checkArgument(
                                                surrogates.containsKey(col),
                                                "Column %s is unacceptable for the Imputer model.",
                                                col));
            }

            Row outputRow = new Row(inputCols.length);
            for (int i = 0; i < inputCols.length; i++) {
                Object value = row.getField(i);
                if (value == null || Double.valueOf(value.toString()).equals(missingValue)) {
                    double surrogate = surrogates.get(inputCols[i]);
                    outputRow.setField(i, surrogate);
                } else {
                    outputRow.setField(i, Double.valueOf(value.toString()));
                }
            }

            return Row.join(row, outputRow);
        }
    }
}
