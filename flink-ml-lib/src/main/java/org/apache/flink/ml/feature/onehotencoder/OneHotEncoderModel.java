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

package org.apache.flink.ml.feature.onehotencoder;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.SparseIntDoubleVectorTypeInfo;
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
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * A Model which encodes data into one-hot format using the model data computed by {@link
 * OneHotEncoder}.
 *
 * <p>The `keep` and `skip` option of {@link HasHandleInvalid} is not supported in {@link
 * OneHotEncoderParams}.
 */
public class OneHotEncoderModel
        implements Model<OneHotEncoderModel>, OneHotEncoderParams<OneHotEncoderModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public OneHotEncoderModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        final String[] inputCols = getInputCols();
        final String[] outputCols = getOutputCols();
        final boolean dropLast = getDropLast();
        final String broadcastModelKey = "OneHotModelStream";

        Preconditions.checkArgument(getHandleInvalid().equals(ERROR_INVALID));
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkArgument(inputCols.length == outputCols.length);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                Collections.nCopies(
                                                outputCols.length,
                                                SparseIntDoubleVectorTypeInfo.INSTANCE)
                                        .toArray(new TypeInformation[0])),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), outputCols));

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);
        DataStream<Tuple2<Integer, Integer>> modelStream =
                OneHotEncoderModelData.getModelDataStream(modelDataTable);

        Function<List<DataStream<?>>, DataStream<Row>> function =
                dataStreams -> {
                    DataStream stream = dataStreams.get(0);
                    return stream.map(
                            new GenerateOutputsFunction(inputCols, dropLast, broadcastModelKey),
                            outputTypeInfo);
                };

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(input),
                        Collections.singletonMap(broadcastModelKey, modelStream),
                        function);

        Table outputTable = tEnv.fromDataStream(output);

        return new Table[] {outputTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                OneHotEncoderModelData.getModelDataStream(modelDataTable),
                path,
                new OneHotEncoderModelData.ModelDataEncoder());
    }

    public static OneHotEncoderModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        OneHotEncoderModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new OneHotEncoderModelData.ModelDataStreamFormat());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public OneHotEncoderModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    private static class GenerateOutputsFunction extends RichMapFunction<Row, Row> {
        private final String[] inputCols;
        private final boolean dropLast;
        private final String broadcastModelKey;
        private List<Tuple2<Integer, Integer>> model = null;

        public GenerateOutputsFunction(
                String[] inputCols, boolean dropLast, String broadcastModelKey) {
            this.inputCols = inputCols;
            this.dropLast = dropLast;
            this.broadcastModelKey = broadcastModelKey;
        }

        @Override
        public Row map(Row row) {
            if (model == null) {
                model = getRuntimeContext().getBroadcastVariable(broadcastModelKey);
            }
            int[] categorySizes = new int[model.size()];
            int offset = dropLast ? 0 : 1;
            for (Tuple2<Integer, Integer> tup : model) {
                categorySizes[tup.f0] = tup.f1 + offset;
            }
            Row result = new Row(categorySizes.length);
            for (int i = 0; i < categorySizes.length; i++) {
                Number number = (Number) row.getField(inputCols[i]);
                Preconditions.checkArgument(
                        number.intValue() == number.doubleValue(),
                        String.format("Value %s cannot be parsed as indexed integer.", number));
                int idx = number.intValue();
                if (idx == categorySizes[i]) {
                    result.setField(i, Vectors.sparse(categorySizes[i], new int[0], new double[0]));
                } else {
                    result.setField(
                            i,
                            Vectors.sparse(categorySizes[i], new int[] {idx}, new double[] {1.0}));
                }
            }

            return Row.join(row, result);
        }
    }
}
