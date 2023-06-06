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

package org.apache.flink.ml.feature.idf;

import org.apache.flink.api.common.functions.RichMapFunction;
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
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/** A Model which transforms data using the model data computed by {@link IDF}. */
public class IDFModel implements Model<IDFModel>, IDFModelParams<IDFModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public IDFModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<Row> data = tEnv.toDataStream(inputs[0]);
        DataStream<IDFModelData> idfModelData = IDFModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        ResolvedSchema schema = inputs[0].getResolvedSchema();
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(schema);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                TableUtils.getTypeInfoByName(schema, getInputCol())),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(data),
                        Collections.singletonMap(broadcastModelKey, idfModelData),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new ComputeTfIdfFunction(broadcastModelKey, getInputCol()),
                                    outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public IDFModel setModelData(Table... inputs) {
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
                IDFModelData.getModelDataStream(modelDataTable),
                path,
                new IDFModelData.ModelDataEncoder());
    }

    public static IDFModel load(StreamTableEnvironment tEnv, String path) throws IOException {
        IDFModel model = ReadWriteUtils.loadStageParam(path);

        Table modelDataTable =
                ReadWriteUtils.loadModelData(tEnv, path, new IDFModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    /** Computes the tf-idf for each term in the input document. */
    private static class ComputeTfIdfFunction extends RichMapFunction<Row, Row> {
        private final String inputCol;
        private final String broadcastKey;
        private DenseIntDoubleVector idf;

        public ComputeTfIdfFunction(String broadcastKey, String inputCol) {
            this.broadcastKey = broadcastKey;
            this.inputCol = inputCol;
        }

        @Override
        public Row map(Row row) {
            if (idf == null) {
                IDFModelData idfModelDataData =
                        (IDFModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                idf = idfModelDataData.idf;
            }

            IntDoubleVector outputVec =
                    ((IntDoubleVector) Objects.requireNonNull(row.getField(inputCol))).clone();
            BLAS.hDot(idf, outputVec);
            return Row.join(row, Row.of(outputVec));
        }
    }
}
