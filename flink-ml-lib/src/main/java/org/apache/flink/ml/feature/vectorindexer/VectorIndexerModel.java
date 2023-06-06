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

package org.apache.flink.ml.feature.vectorindexer;

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
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
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * A Model which encodes input vector to an output vector using the model data computed by {@link
 * VectorIndexer}.
 *
 * <p>The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries in a
 * special bucket, whose index is the number of distinct values in this column.
 */
public class VectorIndexerModel
        implements Model<VectorIndexerModel>, VectorIndexerModelParams<VectorIndexerModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public VectorIndexerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings("unchecked, rawtypes")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String inputCol = getInputCol();
        String outputCol = getOutputCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), outputCol));

        final String broadcastModelKey = "broadcastModelKey";
        DataStream<VectorIndexerModelData> modelDataStream =
                VectorIndexerModelData.getModelDataStream(modelDataTable);

        DataStream<Row> result =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(tEnv.toDataStream(inputs[0])),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            DataStream inputData = inputList.get(0);
                            return inputData.flatMap(
                                    new FindIndex(broadcastModelKey, inputCol, getHandleInvalid()),
                                    outputTypeInfo);
                        });

        return new Table[] {tEnv.fromDataStream(result)};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                VectorIndexerModelData.getModelDataStream(modelDataTable),
                path,
                new VectorIndexerModelData.ModelDataEncoder());
    }

    public static VectorIndexerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        VectorIndexerModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new VectorIndexerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public VectorIndexerModel setModelData(Table... inputs) {
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    /** Finds the index for the input vector using the model data. */
    private static class FindIndex extends RichFlatMapFunction<Row, Row> {
        private final String broadcastModelKey;
        private final String inputCol;
        private final String handleInValid;
        private Map<Integer, Map<Double, Integer>> categoryMaps;

        public FindIndex(String broadcastModelKey, String inputCol, String handleInValid) {
            this.broadcastModelKey = broadcastModelKey;
            this.inputCol = inputCol;
            this.handleInValid = handleInValid;
        }

        @Override
        public void flatMap(Row input, Collector<Row> out) {
            if (categoryMaps == null) {
                VectorIndexerModelData modelData =
                        (VectorIndexerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
                categoryMaps = modelData.categoryMaps;
            }

            IntDoubleVector outputVector = ((IntDoubleVector) input.getField(inputCol)).clone();
            for (Map.Entry<Integer, Map<Double, Integer>> entry : categoryMaps.entrySet()) {
                int columnId = entry.getKey();
                Map<Double, Integer> mapping = entry.getValue();
                double feature = outputVector.get(columnId);
                Integer categoricalFeature = getMapping(feature, mapping, handleInValid);
                if (categoricalFeature == null) {
                    return;
                } else {
                    outputVector.set(columnId, (double) categoricalFeature);
                }
            }

            out.collect(Row.join(input, Row.of(outputVector)));
        }
    }

    /**
     * Maps the input feature to a categorical value using the mappings.
     *
     * @param feature The input continuous feature.
     * @param mapping The mappings from continues features to categorical features.
     * @param handleInValid The way to handle invalid features.
     * @return The categorical value. Returns null if invalid values are skipped.
     */
    private static Integer getMapping(
            double feature, Map<Double, Integer> mapping, String handleInValid) {
        if (mapping.containsKey(feature)) {
            return mapping.get(feature);
        } else {
            switch (handleInValid) {
                case SKIP_INVALID:
                    return null;
                case ERROR_INVALID:
                    throw new RuntimeException(
                            "The input contains unseen double: "
                                    + feature
                                    + ". See "
                                    + HANDLE_INVALID
                                    + " parameter for more options.");
                case KEEP_INVALID:
                    return mapping.size();
                default:
                    throw new UnsupportedOperationException(
                            "Unsupported " + HANDLE_INVALID + "type: " + handleInValid);
            }
        }
    }
}
