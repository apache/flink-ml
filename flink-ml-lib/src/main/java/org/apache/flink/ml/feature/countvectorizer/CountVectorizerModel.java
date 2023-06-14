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

package org.apache.flink.ml.feature.countvectorizer;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/** A Model which transforms data using the model data computed by {@link CountVectorizer}. */
public class CountVectorizerModel
        implements Model<CountVectorizerModel>, CountVectorizerModelParams<CountVectorizerModel> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public CountVectorizerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public CountVectorizerModel setModelData(Table... inputs) {
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
                CountVectorizerModelData.getModelDataStream(modelDataTable),
                path,
                new CountVectorizerModelData.ModelDataEncoder());
    }

    public static CountVectorizerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        CountVectorizerModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new CountVectorizerModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> dataStream = tEnv.toDataStream(inputs[0]);
        DataStream<CountVectorizerModelData> modelDataStream =
                CountVectorizerModelData.getModelDataStream(modelDataTable);

        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                SparseIntDoubleVectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(dataStream),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new PredictOutputFunction(
                                            getInputCol(),
                                            broadcastModelKey,
                                            getMinTF(),
                                            getBinary()),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(output)};
    }

    /** This operator loads model data and predicts result. */
    private static class PredictOutputFunction extends RichMapFunction<Row, Row> {

        private final String inputCol;
        private final String broadcastKey;
        private final double minTF;
        private final boolean binary;
        private Map<String, Integer> vocabulary;

        public PredictOutputFunction(
                String inputCol, String broadcastKey, double minTF, boolean binary) {
            this.inputCol = inputCol;
            this.broadcastKey = broadcastKey;
            this.minTF = minTF;
            this.binary = binary;
        }

        @Override
        public Row map(Row row) throws Exception {
            if (vocabulary == null) {
                CountVectorizerModelData modelData =
                        (CountVectorizerModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                vocabulary = new HashMap<>();
                IntStream.range(0, modelData.vocabulary.length)
                        .forEach(i -> vocabulary.put(modelData.vocabulary[i], i));
            }

            String[] document = (String[]) row.getField(inputCol);
            double[] termCounts = new double[vocabulary.size()];
            for (String word : document) {
                if (vocabulary.containsKey(word)) {
                    termCounts[vocabulary.get(word)] += 1;
                }
            }

            double actualMinTF = minTF >= 1.0 ? minTF : document.length * minTF;
            List<Integer> indices = new ArrayList<>();
            List<Double> values = new ArrayList<>();
            for (int i = 0; i < termCounts.length; i++) {
                if (termCounts[i] >= actualMinTF) {
                    indices.add(i);
                    if (binary) {
                        values.add(1.0);
                    } else {
                        values.add(termCounts[i]);
                    }
                }
            }

            SparseIntDoubleVector outputVec =
                    Vectors.sparse(
                            termCounts.length,
                            indices.stream().mapToInt(i -> i).toArray(),
                            values.stream().mapToDouble(i -> i).toArray());
            return Row.join(row, Row.of(outputVec));
        }
    }
}
