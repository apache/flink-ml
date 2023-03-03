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

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * An Estimator which implements the one-hot encoding algorithm.
 *
 * <p>Data of selected input columns should be indexed numbers in order for OneHotEncoder to
 * function correctly.
 *
 * <p>The `keep` and `skip` option of {@link HasHandleInvalid} is not supported in {@link
 * OneHotEncoderParams}.
 *
 * <p>See https://en.wikipedia.org/wiki/One-hot.
 */
public class OneHotEncoder
        implements Estimator<OneHotEncoder, OneHotEncoderModel>,
                OneHotEncoderParams<OneHotEncoder> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public OneHotEncoder() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OneHotEncoderModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkArgument(getHandleInvalid().equals(ERROR_INVALID));

        final String[] inputCols = getInputCols();

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Integer[]> localMaxIndices =
                tEnv.toDataStream(inputs[0])
                        .transform(
                                "ExtractInputValueAndFindMaxIndexOperator",
                                ObjectArrayTypeInfo.getInfoFor(BasicTypeInfo.INT_TYPE_INFO),
                                new ExtractInputValueAndFindMaxIndexOperator(inputCols));

        DataStream<Tuple2<Integer, Integer>> modelData =
                localMaxIndices
                        .transform(
                                "GenerateModelDataOperator",
                                TupleTypeInfo.getBasicTupleTypeInfo(Integer.class, Integer.class),
                                new GenerateModelDataOperator(inputCols.length))
                        .setParallelism(1);

        OneHotEncoderModel model =
                new OneHotEncoderModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static OneHotEncoder load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Operator to extract the integer values from input columns and to find the max index value for
     * each column.
     */
    private static class ExtractInputValueAndFindMaxIndexOperator
            extends AbstractStreamOperator<Integer[]>
            implements OneInputStreamOperator<Row, Integer[]>, BoundedOneInput {

        private final String[] inputCols;

        private ListState<Integer[]> maxIndicesState;

        private Integer[] maxIndices;

        private ExtractInputValueAndFindMaxIndexOperator(String[] inputCols) {
            this.inputCols = inputCols;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            TypeInformation<Integer[]> type =
                    ObjectArrayTypeInfo.getInfoFor(BasicTypeInfo.INT_TYPE_INFO);

            maxIndicesState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("maxIndices", type));

            maxIndices =
                    OperatorStateUtils.getUniqueElement(maxIndicesState, "maxIndices")
                            .orElse(initMaxIndices(inputCols.length));
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            maxIndicesState.update(Collections.singletonList(maxIndices));
        }

        @Override
        public void processElement(StreamRecord<Row> streamRecord) {
            Row row = streamRecord.getValue();
            for (int i = 0; i < inputCols.length; i++) {
                Number number = (Number) row.getField(inputCols[i]);
                int value = number.intValue();

                if (value != number.doubleValue()) {
                    throw new IllegalArgumentException(
                            String.format("Value %s cannot be parsed as indexed integer.", number));
                }
                Preconditions.checkArgument(value >= 0, "Negative value not supported.");

                if (value > maxIndices[i]) {
                    maxIndices[i] = value;
                }
            }
        }

        @Override
        public void endInput() {
            output.collect(new StreamRecord<>(maxIndices));
        }
    }

    /**
     * Collects and reduces the max index value in each column and produces the model data.
     *
     * <p>Output: Pairs of column index and max index value in this column.
     */
    private static class GenerateModelDataOperator
            extends AbstractStreamOperator<Tuple2<Integer, Integer>>
            implements OneInputStreamOperator<Integer[], Tuple2<Integer, Integer>>,
                    BoundedOneInput {
        private final int inputColsNum;

        private ListState<Integer[]> maxIndicesState;

        private Integer[] maxIndices;

        private GenerateModelDataOperator(int inputColsNum) {
            this.inputColsNum = inputColsNum;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            TypeInformation<Integer[]> type =
                    ObjectArrayTypeInfo.getInfoFor(BasicTypeInfo.INT_TYPE_INFO);

            maxIndicesState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("maxIndices", type));

            maxIndices =
                    OperatorStateUtils.getUniqueElement(maxIndicesState, "maxIndices")
                            .orElse(initMaxIndices(inputColsNum));
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            maxIndicesState.update(Collections.singletonList(maxIndices));
        }

        @Override
        public void processElement(StreamRecord<Integer[]> streamRecord) {
            Integer[] indices = streamRecord.getValue();
            for (int i = 0; i < maxIndices.length; i++) {
                if (indices[i] > maxIndices[i]) {
                    maxIndices[i] = indices[i];
                }
            }
        }

        @Override
        public void endInput() {
            for (int i = 0; i < maxIndices.length; i++) {
                output.collect(new StreamRecord<>(Tuple2.of(i, maxIndices[i])));
            }
        }
    }

    private static Integer[] initMaxIndices(int length) {
        Integer[] indices = new Integer[length];
        Arrays.fill(indices, Integer.MIN_VALUE);
        return indices;
    }
}
