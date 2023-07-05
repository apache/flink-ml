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
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.RowUtils;
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
 * A Model which transforms input index column(s) to string column(s) using the model data computed
 * by {@link StringIndexer}. It is a reverse operation of {@link StringIndexerModel}.
 */
public class IndexToStringModel
        implements Model<IndexToStringModel>, IndexToStringModelParams<IndexToStringModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public IndexToStringModel() {
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

    public static IndexToStringModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        IndexToStringModel model = ReadWriteUtils.loadStageParam(path);
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
    public IndexToStringModel setModelData(Table... inputs) {
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
        Arrays.fill(outputTypes, BasicTypeInfo.STRING_TYPE_INFO);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), outputTypes),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), outputCols));

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
                                    new Index2String(broadcastModelKey, inputCols), outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(result)};
    }

    /** Maps the input index values to string values according to the model data. */
    private static class Index2String extends RichFlatMapFunction<Row, Row> {
        private String[][] stringArrays;
        private final String broadcastModelKey;
        private final String[] inputCols;

        public Index2String(String broadcastModelKey, String[] inputCols) {
            this.broadcastModelKey = broadcastModelKey;
            this.inputCols = inputCols;
        }

        @Override
        public void flatMap(Row input, Collector<Row> out) {
            if (stringArrays == null) {
                StringIndexerModelData modelData =
                        (StringIndexerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                stringArrays = modelData.stringArrays;
            }

            Row result = RowUtils.cloneWithReservedFields(input, inputCols.length);
            for (int i = 0; i < inputCols.length; i++) {
                int stringId = (Integer) input.getField(inputCols[i]);
                if (stringId < stringArrays[i].length && stringId >= 0) {
                    result.setField(i + input.getArity(), stringArrays[i][stringId]);
                } else {
                    throw new RuntimeException(
                            "The input contains unseen index: " + stringId + ".");
                }
            }

            out.collect(result);
        }
    }
}
