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

package org.apache.flink.ml.feature.bucketizer;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A Transformer that maps multiple columns of continuous features to multiple columns of discrete
 * features, i.e., buckets indices. The indices are in [0, numSplitsInThisColumn - 1].
 *
 * <p>The `keep` option of {@link HasHandleInvalid} means that we put the invalid data in the last
 * bucket of the splits, whose index is the number of the buckets.
 */
public class Bucketizer implements Transformer<Bucketizer>, BucketizerParams<Bucketizer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Bucketizer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        String[] inputCols = getInputCols();
        String[] outputCols = getOutputCols();
        Double[][] splitsArray = getSplitsArray();
        Preconditions.checkArgument(inputCols.length == outputCols.length);
        Preconditions.checkArgument(inputCols.length == splitsArray.length);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        TypeInformation<?>[] outputTypes = new TypeInformation[outputCols.length];
        Arrays.fill(outputTypes, BasicTypeInfo.DOUBLE_TYPE_INFO);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), outputTypes),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCols()));

        int[] inputColumnIndexes =
                TableUtils.getColumnIndexes(inputs[0].getResolvedSchema(), inputCols);
        DataStream<Row> result =
                tEnv.toDataStream(inputs[0])
                        .flatMap(
                                new FindBucketFunction(
                                        inputColumnIndexes, splitsArray, getHandleInvalid()),
                                outputTypeInfo);
        return new Table[] {tEnv.fromDataStream(result)};
    }

    /** Finds the bucket index for each continuous feature of an input data point. */
    private static class FindBucketFunction implements FlatMapFunction<Row, Row> {
        private final int[] inputCols;
        private final String handleInvalid;
        private final Double[][] splitsArray;

        public FindBucketFunction(int[] inputCols, Double[][] splitsArray, String handleInvalid) {
            this.inputCols = inputCols;
            this.splitsArray = splitsArray;
            this.handleInvalid = handleInvalid;
        }

        @Override
        public void flatMap(Row value, Collector<Row> out) {
            Row outputRow = new Row(inputCols.length);

            for (int i = 0; i < inputCols.length; i++) {
                double feature = ((Number) value.getField(inputCols[i])).doubleValue();
                Double[] splits = splitsArray[i];
                boolean isInvalid = false;

                if (!Double.isNaN(feature)) {
                    double index = Arrays.binarySearch(splits, feature);
                    if (index >= 0) {
                        if (index == splits.length - 1) {
                            index--;
                        }
                        outputRow.setField(i, index);
                    } else {
                        index = -index - 1;
                        if (index == 0 || index == splits.length) {
                            isInvalid = true;
                        } else {
                            outputRow.setField(i, index - 1);
                        }
                    }
                } else {
                    isInvalid = true;
                }

                if (isInvalid) {
                    switch (handleInvalid) {
                        case ERROR_INVALID:
                            throw new RuntimeException(
                                    "The input contains invalid value. See "
                                            + HANDLE_INVALID
                                            + " parameter for more options.");
                        case SKIP_INVALID:
                            return;
                        case KEEP_INVALID:
                            outputRow.setField(i, (double) splits.length - 1);
                            break;
                        default:
                            throw new UnsupportedOperationException(
                                    "Unsupported " + HANDLE_INVALID + " type: " + handleInvalid);
                    }
                }
            }
            out.collect(Row.join(value, outputRow));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Bucketizer load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
