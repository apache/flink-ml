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

package org.apache.flink.ml.feature.kbinsdiscretizer;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
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
 * A Model which transforms continuous features into discrete features using the model data computed
 * by {@link KBinsDiscretizer}.
 *
 * <p>A feature value `v` should be mapped to a bin with edges as `{left, right}` if `v` is in
 * `[left, right)`. If `v` does not fall into any of the bins, it is mapped to the closest bin. For
 * example suppose the bin edges are `{-1, 0, 1}` for one column, then we have two bins `{-1, 0}`
 * and `{0, 1}`. In this case, -2 is mapped into 0-th bin, 0 is mapped into the 1-st bin and 2 is
 * mapped into the 1-st bin.
 */
public class KBinsDiscretizerModel
        implements Model<KBinsDiscretizerModel>,
                KBinsDiscretizerModelParams<KBinsDiscretizerModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public KBinsDiscretizerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<Row> inputData = tEnv.toDataStream(inputs[0]);
        DataStream<KBinsDiscretizerModelData> modelData =
                KBinsDiscretizerModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                TypeInformation.of(DenseIntDoubleVector.class)),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputData),
                        Collections.singletonMap(broadcastModelKey, modelData),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new FindBinFunction(getInputCol(), broadcastModelKey),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public KBinsDiscretizerModel setModelData(Table... inputs) {
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
                KBinsDiscretizerModelData.getModelDataStream(modelDataTable),
                path,
                new KBinsDiscretizerModelData.ModelDataEncoder());
    }

    public static KBinsDiscretizerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        KBinsDiscretizerModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new KBinsDiscretizerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    private static class FindBinFunction extends RichMapFunction<Row, Row> {
        private final String inputCol;
        private final String broadcastKey;
        /** Model data used to find bins for each feature. */
        private double[][] binEdges;

        public FindBinFunction(String inputCol, String broadcastKey) {
            this.inputCol = inputCol;
            this.broadcastKey = broadcastKey;
        }

        @Override
        public Row map(Row row) {
            if (binEdges == null) {
                KBinsDiscretizerModelData modelData =
                        (KBinsDiscretizerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                binEdges = modelData.binEdges;
            }
            DenseIntDoubleVector inputVec = ((IntDoubleVector) row.getField(inputCol)).toDense();
            DenseIntDoubleVector outputVec = inputVec.clone();
            for (int i = 0; i < inputVec.size(); i++) {
                double targetFeature = inputVec.get(i);
                int index = Arrays.binarySearch(binEdges[i], targetFeature);
                if (index < 0) {
                    // Computes the index to insert.
                    index = -index - 1;
                    // Puts it in the left bin.
                    index--;
                }
                // Handles the boundary.
                index = Math.min(index, (binEdges[i].length - 2));
                index = Math.max(index, 0);

                outputVec.set(i, (double) index);
            }
            return Row.join(row, Row.of(outputVec));
        }
    }
}
