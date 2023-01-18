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

package org.apache.flink.ml.feature.stringindexer;

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * A Model which transforms input string/numeric column(s) to double column(s) using the model data
 * computed by {@link StringIndexer}.
 *
 * <p>The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries in a
 * special bucket, whose index is the number of distinct values in this column.
 */
public class StringIndexerModel
        implements Model<StringIndexerModel>, StringIndexerModelParams<StringIndexerModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public StringIndexerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                StringIndexerModelData.getModelDataStream(modelDataTable),
                path,
                new StringIndexerModelData.ModelDataEncoder());
    }

    public static StringIndexerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        StringIndexerModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new StringIndexerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public StringIndexerModel setModelData(Table... inputs) {
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    @SuppressWarnings("unchecked, rawtypes")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String[] inputCols = getInputCols();
        String[] outputCols = getOutputCols();
        Preconditions.checkArgument(inputCols.length == outputCols.length);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        TypeInformation<?>[] outputTypes = new TypeInformation[outputCols.length];
        Arrays.fill(outputTypes, BasicTypeInfo.DOUBLE_TYPE_INFO);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), outputTypes),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCols()));

        final String broadcastModelKey = "broadcastModelKey";
        DataStream<StringIndexerModelData> modelDataStream =
                StringIndexerModelData.getModelDataStream(modelDataTable);

        DataStream<Row> result =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(tEnv.toDataStream(inputs[0])),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            DataStream inputData = inputList.get(0);
                            return inputData.flatMap(
                                    new String2Index(
                                            broadcastModelKey, inputCols, getHandleInvalid()),
                                    outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(result)};
    }

    /** Maps the input columns to double values according to the model data. */
    private static class String2Index extends RichFlatMapFunction<Row, Row> {
        private HashMap<String, Double>[] modelDataMap;
        private final String broadcastModelKey;
        private final String[] inputCols;
        private final String handleInValid;

        public String2Index(String broadcastModelKey, String[] inputCols, String handleInValid) {
            this.broadcastModelKey = broadcastModelKey;
            this.inputCols = inputCols;
            this.handleInValid = handleInValid;
        }

        @Override
        @SuppressWarnings("unchecked")
        public void flatMap(Row input, Collector<Row> out) {
            if (modelDataMap == null) {
                modelDataMap = new HashMap[inputCols.length];
                StringIndexerModelData modelData =
                        (StringIndexerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                String[][] stringsArray = modelData.stringArrays;
                for (int i = 0; i < stringsArray.length; i++) {
                    double idx = 0.0;
                    modelDataMap[i] = new HashMap<>(stringsArray[i].length);
                    for (String string : stringsArray[i]) {
                        modelDataMap[i].put(string, idx++);
                    }
                }
            }

            Row outputIndices = new Row(inputCols.length);
            for (int i = 0; i < inputCols.length; i++) {
                Object objVal = input.getField(inputCols[i]);
                String stringVal;
                if (null == objVal) {
                    stringVal = null;
                } else if (objVal instanceof String) {
                    stringVal = (String) objVal;
                } else if (objVal instanceof Number) {
                    stringVal = String.valueOf(objVal);
                } else {
                    throw new RuntimeException(
                            "The input column only supports string and numeric type.");
                }

                if (modelDataMap[i].containsKey(stringVal)) {
                    outputIndices.setField(i, modelDataMap[i].get(stringVal));
                } else {
                    switch (handleInValid) {
                        case SKIP_INVALID:
                            return;
                        case ERROR_INVALID:
                            throw new RuntimeException(
                                    "The input contains unseen string: "
                                            + stringVal
                                            + ". See "
                                            + HANDLE_INVALID
                                            + " parameter for more options.");
                        case KEEP_INVALID:
                            outputIndices.setField(i, (double) modelDataMap[i].size());
                            break;
                        default:
                            throw new UnsupportedOperationException(
                                    "Unsupported " + HANDLE_INVALID + "type: " + handleInValid);
                    }
                }
            }

            out.collect(Row.join(input, outputIndices));
        }
    }
}
