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

package org.apache.flink.ml.feature.vectorslicer;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A Transformer that transforms a vector to a new feature, which is a sub-array of the original
 * feature. It is useful for extracting features from a given vector.
 *
 * <p>Note that duplicate features are not allowed, so there can be no overlap between selected
 * indices. If the max value of the indices is greater than the size of the input vector, it throws
 * an IllegalArgumentException.
 */
public class VectorSlicer implements Transformer<VectorSlicer>, VectorSlicerParams<VectorSlicer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public VectorSlicer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));
        DataStream<Row> output =
                tEnv.toDataStream(inputs[0])
                        .map(new VectorSliceFunction(getIndices(), getInputCol()), outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);
        return new Table[] {outputTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static VectorSlicer load(StreamTableEnvironment env, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Vector slice function which transforms a vector to a new one with a sub-array of the original
     * features.
     */
    private static class VectorSliceFunction implements MapFunction<Row, Row> {
        private final Integer[] indices;
        private final String inputCol;
        private int maxIndex = -1;

        public VectorSliceFunction(Integer[] indices, String inputCol) {
            this.indices = indices;
            for (Integer index : indices) {
                maxIndex = Math.max(maxIndex, index);
            }
            this.inputCol = inputCol;
        }

        @Override
        public Row map(Row row) throws Exception {
            IntDoubleVector inputVec = row.getFieldAs(inputCol);
            IntDoubleVector outputVec;
            if (maxIndex >= inputVec.size()) {
                throw new IllegalArgumentException(
                        "Index value "
                                + maxIndex
                                + " is greater than vector size:"
                                + inputVec.size());
            }
            if (inputVec instanceof DenseIntDoubleVector) {
                double[] values = new double[indices.length];
                for (int i = 0; i < indices.length; ++i) {
                    values[i] = ((DenseIntDoubleVector) inputVec).values[indices[i]];
                }
                outputVec = new DenseIntDoubleVector(values);
            } else {
                int nnz = 0;
                SparseIntDoubleVector vec = (SparseIntDoubleVector) inputVec;
                int[] outputIndices = new int[indices.length];
                double[] outputValues = new double[indices.length];
                for (int i = 0; i < indices.length; i++) {
                    double val = vec.get(indices[i]);
                    if (val != 0) {
                        outputIndices[nnz] = i;
                        outputValues[nnz] = val;
                        nnz++;
                    }
                }
                if (nnz < outputIndices.length) {
                    outputIndices = Arrays.copyOf(outputIndices, nnz);
                    outputValues = Arrays.copyOf(outputValues, nnz);
                }
                outputVec = new SparseIntDoubleVector(indices.length, outputIndices, outputValues);
            }
            return Row.join(row, Row.of(outputVec));
        }
    }
}
