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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the string indexing algorithm.
 *
 * <p>A string indexer maps one or more columns (string/numerical value) of the input to one or more
 * indexed output columns (integer value). The output indices of two data points are the same iff
 * their corresponding input columns are the same. The indices are in [0,
 * numDistinctValuesInThisColumn].
 *
 * <p>The input columns are cast to string if they are numeric values. By default, the output model
 * is arbitrarily ordered. Users can control this by setting {@link
 * StringIndexerParams#STRING_ORDER_TYPE}.
 */
public class StringIndexer
        implements Estimator<StringIndexer, StringIndexerModel>,
                StringIndexerParams<StringIndexer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public StringIndexer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static StringIndexer load(StreamExecutionEnvironment env, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public StringIndexerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String[] inputCols = getInputCols();
        String[] outputCols = getOutputCols();
        Preconditions.checkArgument(inputCols.length == outputCols.length);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<Tuple2<Integer, String>> columnIdAndString =
                tEnv.toDataStream(inputs[0]).flatMap(new ExtractColumnIdAndString(inputCols));

        DataStream<Tuple3<Integer, String, Long>> columnIdAndStringAndCnt =
                DataStreamUtils.mapPartition(
                        columnIdAndString.keyBy(
                                (KeySelector<Tuple2<Integer, String>, Integer>) Tuple2::hashCode),
                        new CountStringsByColumn(inputCols.length));

        DataStream<StringIndexerModelData> modelData =
                DataStreamUtils.mapPartition(
                        columnIdAndStringAndCnt,
                        new GenerateModel(inputCols.length, getStringOrderType()));
        modelData.getTransformation().setParallelism(1);

        StringIndexerModel model =
                new StringIndexerModel().setModelData(tEnv.fromDataStream(modelData));
        ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /**
     * Merges all the extracted strings and generates the {@link StringIndexerModelData} according
     * to the specified string order type.
     */
    private static class GenerateModel
            implements MapPartitionFunction<Tuple3<Integer, String, Long>, StringIndexerModelData> {
        private final int numCols;
        private final String stringOrderType;

        public GenerateModel(int numCols, String stringOrderType) {
            this.numCols = numCols;
            this.stringOrderType = stringOrderType;
        }

        @Override
        @SuppressWarnings("unchecked")
        public void mapPartition(
                Iterable<Tuple3<Integer, String, Long>> values,
                Collector<StringIndexerModelData> out) {
            String[][] stringArrays = new String[numCols][];
            ArrayList<Tuple2<String, Long>>[] stringsAndCntsByColumn = new ArrayList[numCols];
            for (int i = 0; i < numCols; i++) {
                stringsAndCntsByColumn[i] = new ArrayList<>();
            }

            for (Tuple3<Integer, String, Long> colIdAndStringAndCnt : values) {
                stringsAndCntsByColumn[colIdAndStringAndCnt.f0].add(
                        Tuple2.of(colIdAndStringAndCnt.f1, colIdAndStringAndCnt.f2));
            }

            for (int i = 0; i < stringsAndCntsByColumn.length; i++) {
                List<Tuple2<String, Long>> stringsAndCnts = stringsAndCntsByColumn[i];
                switch (stringOrderType) {
                    case StringIndexerParams.ALPHABET_ASC_ORDER:
                        stringsAndCnts.sort(Comparator.comparing(valAndCnt -> valAndCnt.f0));
                        break;
                    case StringIndexerParams.ALPHABET_DESC_ORDER:
                        stringsAndCnts.sort(
                                (valAndCnt1, valAndCnt2) ->
                                        -valAndCnt1.f0.compareTo(valAndCnt2.f0));
                        break;
                    case StringIndexerParams.FREQUENCY_ASC_ORDER:
                        stringsAndCnts.sort(Comparator.comparing(valAndCnt -> valAndCnt.f1));
                        break;
                    case StringIndexerParams.FREQUENCY_DESC_ORDER:
                        stringsAndCnts.sort(
                                (valAndCnt1, valAndCnt2) ->
                                        -valAndCnt1.f1.compareTo(valAndCnt2.f1));
                        break;
                    case StringIndexerParams.ARBITRARY_ORDER:
                        break;
                    default:
                        throw new IllegalStateException(
                                "Unsupported string order type: " + stringOrderType);
                }

                stringArrays[i] = new String[stringsAndCnts.size()];
                for (int stringId = 0; stringId < stringArrays[i].length; stringId++) {
                    stringArrays[i][stringId] = stringsAndCnts.get(stringId).f0;
                }
            }

            out.collect(new StringIndexerModelData(stringArrays));
        }
    }

    /** Computes the frequency of strings in each column. */
    private static class CountStringsByColumn
            implements MapPartitionFunction<
                    Tuple2<Integer, String>, Tuple3<Integer, String, Long>> {
        private final int numCols;

        public CountStringsByColumn(int numCols) {
            this.numCols = numCols;
        }

        @Override
        @SuppressWarnings("unchecked")
        public void mapPartition(
                Iterable<Tuple2<Integer, String>> values,
                Collector<Tuple3<Integer, String, Long>> out) {
            HashMap<String, Long>[] string2CntByColumn = new HashMap[numCols];
            for (int i = 0; i < numCols; i++) {
                string2CntByColumn[i] = new HashMap<>();
            }
            for (Tuple2<Integer, String> columnIdAndString : values) {
                int colId = columnIdAndString.f0;
                String stringVal = columnIdAndString.f1;
                long cnt = string2CntByColumn[colId].getOrDefault(stringVal, 0L) + 1;
                string2CntByColumn[colId].put(stringVal, cnt);
            }
            for (int i = 0; i < numCols; i++) {
                for (Map.Entry<String, Long> entry : string2CntByColumn[i].entrySet()) {
                    out.collect(Tuple3.of(i, entry.getKey(), entry.getValue()));
                }
            }
        }
    }

    /** Extracts strings by column. */
    private static class ExtractColumnIdAndString
            implements FlatMapFunction<Row, Tuple2<Integer, String>> {
        private final String[] inputCols;

        public ExtractColumnIdAndString(String[] inputCols) {
            this.inputCols = inputCols;
        }

        @Override
        public void flatMap(Row row, Collector<Tuple2<Integer, String>> out) {
            for (int i = 0; i < inputCols.length; i++) {
                Object objVal = row.getField(inputCols[i]);
                String stringVal;
                if (objVal instanceof String) {
                    stringVal = (String) objVal;
                } else if (objVal instanceof Number) {
                    stringVal = String.valueOf(objVal);
                } else {
                    throw new RuntimeException(
                            "The input column only supports string and numeric type.");
                }
                out.collect(Tuple2.of(i, stringVal));
            }
        }
    }
}
