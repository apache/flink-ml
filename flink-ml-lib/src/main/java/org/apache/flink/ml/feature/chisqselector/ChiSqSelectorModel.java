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

package org.apache.flink.ml.feature.chisqselector;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A Model that selects features using the model data computed by {@link ChiSqSelector}. */
public class ChiSqSelectorModel
        implements Model<ChiSqSelectorModel>, ChiSqSelectorModelParams<ChiSqSelectorModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public ChiSqSelectorModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<Row> inputData = tEnv.toDataStream(inputs[0]);
        DataStream<ChiSqSelectorModelData> modelData =
                ChiSqSelectorModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputData),
                        Collections.singletonMap(broadcastModelKey, modelData),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new SelectFeaturesFunction(getFeaturesCol(), broadcastModelKey),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public ChiSqSelectorModel setModelData(Table... inputs) {
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
                ChiSqSelectorModelData.getModelDataStream(modelDataTable),
                path,
                new ChiSqSelectorModelData.ModelDataEncoder());
    }

    public static ChiSqSelectorModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        ChiSqSelectorModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new ChiSqSelectorModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * A map function that selects features from input vector according to the model data generated
     * by {@link ChiSqSelector}.
     */
    private static class SelectFeaturesFunction extends RichMapFunction<Row, Row> {
        private final String featuresCol;
        private final String broadcastKey;
        private int[] selectedFeatureIndices;

        private SelectFeaturesFunction(String featuresCol, String broadcastKey) {
            this.featuresCol = featuresCol;
            this.broadcastKey = broadcastKey;
        }

        @Override
        public Row map(Row row) throws Exception {
            if (selectedFeatureIndices == null) {
                ChiSqSelectorModelData modelData =
                        (ChiSqSelectorModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                selectedFeatureIndices = modelData.selectedFeatureIndices;
            }

            Vector inputVector = row.getFieldAs(featuresCol);
            Vector outputVector;
            if (inputVector instanceof SparseVector) {
                List<Integer> outputIndices = new ArrayList<>();
                List<Double> outputValues = new ArrayList<>();
                int[] inputIndices = ((SparseVector) inputVector).indices;
                double[] inputValues = ((SparseVector) inputVector).values;
                int inputIndex = 0;
                int featureIndex = 0;
                while (featureIndex < selectedFeatureIndices.length
                        && inputIndex < inputIndices.length) {
                    if (selectedFeatureIndices[featureIndex] < inputIndices[inputIndex]) {
                        featureIndex++;
                    } else if (selectedFeatureIndices[featureIndex] > inputIndices[inputIndex]) {
                        inputIndex++;
                    } else {
                        if (inputValues[inputIndex] != 0) {
                            outputIndices.add(featureIndex);
                            outputValues.add(inputValues[inputIndex]);
                        }
                        featureIndex++;
                        inputIndex++;
                    }
                }
                outputVector =
                        new SparseVector(
                                selectedFeatureIndices.length,
                                outputIndices.stream().mapToInt(x -> x).toArray(),
                                outputValues.stream().mapToDouble(x -> x).toArray());
            } else {
                outputVector = new DenseVector(selectedFeatureIndices.length);
                for (int i = 0; i < selectedFeatureIndices.length; i++) {
                    outputVector.set(i, inputVector.get(selectedFeatureIndices[i]));
                }
            }

            return Row.join(row, Row.of(outputVector));
        }
    }
}
