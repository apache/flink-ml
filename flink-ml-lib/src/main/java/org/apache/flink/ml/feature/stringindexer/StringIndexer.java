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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

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
 *
 * <p>The `keep` option of {@link HasHandleInvalid} means that we put the invalid entries in a
 * special bucket, whose index is the number of distinct values in this column.
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

    public static StringIndexer load(StreamTableEnvironment tEnv, String path) throws IOException {
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

        DataStream<Map<String, Long>[]> localCountedString =
                tEnv.toDataStream(inputs[0])
                        .transform(
                                "countStringOperator",
                                Types.OBJECT_ARRAY(Types.MAP(Types.STRING, Types.LONG)),
                                new CountStringOperator(inputCols));

        DataStream<Map<String, Long>[]> countedString =
                DataStreamUtils.reduce(
                        localCountedString,
                        (ReduceFunction<Map<String, Long>[]>)
                                (value1, value2) -> {
                                    for (int i = 0; i < value1.length; i++) {
                                        for (Entry<String, Long> stringAndCnt :
                                                value2[i].entrySet()) {
                                            value1[i].compute(
                                                    stringAndCnt.getKey(),
                                                    (k, v) ->
                                                            (v == null
                                                                    ? stringAndCnt.getValue()
                                                                    : v + stringAndCnt.getValue()));
                                        }
                                    }
                                    return value1;
                                },
                        Types.OBJECT_ARRAY(Types.MAP(Types.STRING, Types.LONG)));

        DataStream<StringIndexerModelData> modelData =
                countedString.map(new ModelGenerator(getStringOrderType()));
        modelData.getTransformation().setParallelism(1);

        StringIndexerModel model =
                new StringIndexerModel().setModelData(tEnv.fromDataStream(modelData));
        ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /** Computes the occurrence time of each string by columns. */
    private static class CountStringOperator extends AbstractStreamOperator<Map<String, Long>[]>
            implements OneInputStreamOperator<Row, Map<String, Long>[]>, BoundedOneInput {
        /** The name of input columns. */
        private final String[] inputCols;
        /** The occurrence time of each string by column. */
        private Map<String, Long>[] stringCntByColumn;
        /** The state of stringCntByColumn. */
        private ListState<Map<String, Long>[]> stringCntByColumnState;

        public CountStringOperator(String[] inputCols) {
            this.inputCols = inputCols;
            stringCntByColumn = new HashMap[inputCols.length];
            for (int i = 0; i < stringCntByColumn.length; i++) {
                stringCntByColumn[i] = new HashMap<>();
            }
        }

        @Override
        public void endInput() {
            output.collect(new StreamRecord<>(stringCntByColumn));
            stringCntByColumnState.clear();
        }

        @Override
        public void processElement(StreamRecord<Row> element) {
            Row r = element.getValue();
            for (int i = 0; i < inputCols.length; i++) {
                Object objVal = r.getField(inputCols[i]);
                String stringVal;
                if (null == objVal) {
                    // Null values should be ignored.
                    continue;
                } else if (objVal instanceof String) {
                    stringVal = (String) objVal;
                } else if (objVal instanceof Number) {
                    stringVal = String.valueOf(objVal);
                } else {
                    throw new RuntimeException(
                            "The input column only supports string and numeric type.");
                }
                stringCntByColumn[i].compute(stringVal, (k, v) -> (v == null ? 1 : v + 1));
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            stringCntByColumnState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "stringCntByColumnState",
                                            Types.OBJECT_ARRAY(
                                                    Types.MAP(Types.STRING, Types.LONG))));

            OperatorStateUtils.getUniqueElement(stringCntByColumnState, "stringCntByColumnState")
                    .ifPresent(x -> stringCntByColumn = x);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            stringCntByColumnState.update(Collections.singletonList(stringCntByColumn));
        }
    }

    /**
     * Merges all the extracted strings and generates the {@link StringIndexerModelData} according
     * to the specified string order type.
     */
    private static class ModelGenerator
            implements MapFunction<Map<String, Long>[], StringIndexerModelData> {
        private final String stringOrderType;

        public ModelGenerator(String stringOrderType) {
            this.stringOrderType = stringOrderType;
        }

        @Override
        public StringIndexerModelData map(Map<String, Long>[] value) {
            int numCols = value.length;
            String[][] stringArrays = new String[numCols][];
            ArrayList<Tuple2<String, Long>> stringsAndCnts = new ArrayList<>();
            for (int i = 0; i < numCols; i++) {
                stringsAndCnts.clear();
                stringsAndCnts.ensureCapacity(value[i].size());
                for (Map.Entry<String, Long> entry : value[i].entrySet()) {
                    stringsAndCnts.add(Tuple2.of(entry.getKey(), entry.getValue()));
                }
                switch (stringOrderType) {
                    case ALPHABET_ASC_ORDER:
                        stringsAndCnts.sort(Comparator.comparing(valAndCnt -> valAndCnt.f0));
                        break;
                    case ALPHABET_DESC_ORDER:
                        stringsAndCnts.sort(
                                (valAndCnt1, valAndCnt2) ->
                                        -valAndCnt1.f0.compareTo(valAndCnt2.f0));
                        break;
                    case FREQUENCY_ASC_ORDER:
                        stringsAndCnts.sort(Comparator.comparing(valAndCnt -> valAndCnt.f1));
                        break;
                    case FREQUENCY_DESC_ORDER:
                        stringsAndCnts.sort(
                                (valAndCnt1, valAndCnt2) ->
                                        -valAndCnt1.f1.compareTo(valAndCnt2.f1));
                        break;
                    case ARBITRARY_ORDER:
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                "Unsupported "
                                        + STRING_ORDER_TYPE
                                        + " type: "
                                        + stringOrderType
                                        + ".");
                }
                stringArrays[i] = stringsAndCnts.stream().map(x -> x.f0).toArray(String[]::new);
            }

            return new StringIndexerModelData(stringArrays);
        }
    }
}
