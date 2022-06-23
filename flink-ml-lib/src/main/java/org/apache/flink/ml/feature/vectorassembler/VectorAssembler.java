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

package org.apache.flink.ml.feature.vectorassembler;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
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
import java.util.HashMap;
import java.util.Map;

/**
 * A feature transformer that combines a given list of input columns into a vector column. Types of
 * input columns must be either vector or numerical value.
 *
 * <p>The `keep` option of {@link HasHandleInvalid} means that we output bad rows with output column
 * set to null.
 */
public class VectorAssembler
        implements Transformer<VectorAssembler>, VectorAssemblerParams<VectorAssembler> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final double RATIO = 1.5;

    public VectorAssembler() {
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
                        .flatMap(
                                new AssemblerFunc(getInputCols(), getHandleInvalid()),
                                outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);
        return new Table[] {outputTable};
    }

    private static class AssemblerFunc implements FlatMapFunction<Row, Row> {
        private final String[] inputCols;
        private final String handleInvalid;

        public AssemblerFunc(String[] inputCols, String handleInvalid) {
            this.inputCols = inputCols;
            this.handleInvalid = handleInvalid;
        }

        @Override
        public void flatMap(Row value, Collector<Row> out) {
            int nnz = 0;
            int vectorSize = 0;
            try {
                for (String inputCol : inputCols) {
                    Object object = value.getField(inputCol);
                    Preconditions.checkNotNull(object, "Input column value should not be null.");
                    if (object instanceof Number) {
                        nnz += 1;
                        vectorSize += 1;
                    } else if (object instanceof SparseVector) {
                        nnz += ((SparseVector) object).indices.length;
                        vectorSize += ((SparseVector) object).size();
                    } else if (object instanceof DenseVector) {
                        nnz += ((DenseVector) object).size();
                        vectorSize += ((DenseVector) object).size();
                    } else {
                        throw new IllegalArgumentException(
                                "Input type has not been supported yet.");
                    }
                }
            } catch (Exception e) {
                switch (handleInvalid) {
                    case ERROR_INVALID:
                        throw e;
                    case SKIP_INVALID:
                        return;
                    case KEEP_INVALID:
                        out.collect(Row.join(value, Row.of((Object) null)));
                        return;
                    default:
                        throw new UnsupportedOperationException(
                                "Unsupported " + HANDLE_INVALID + " type: " + handleInvalid);
                }
            }

            boolean toDense = nnz * RATIO > vectorSize;
            Vector assembledVec =
                    toDense
                            ? assembleDense(inputCols, value, vectorSize)
                            : assembleSparse(inputCols, value, vectorSize, nnz);
            out.collect(Row.join(value, Row.of(assembledVec)));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static VectorAssembler load(StreamTableEnvironment env, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /** Assembles the input columns into a dense vector. */
    private static Vector assembleDense(String[] inputCols, Row inputRow, int vectorSize) {
        double[] values = new double[vectorSize];
        int currentOffset = 0;

        for (String inputCol : inputCols) {
            Object object = inputRow.getField(inputCol);
            if (object instanceof Number) {
                values[currentOffset++] = ((Number) object).doubleValue();
            } else if (object instanceof SparseVector) {
                SparseVector sparseVector = (SparseVector) object;
                for (int i = 0; i < sparseVector.indices.length; i++) {
                    values[currentOffset + sparseVector.indices[i]] = sparseVector.values[i];
                }
                currentOffset += sparseVector.size();

            } else {
                DenseVector denseVector = (DenseVector) object;
                System.arraycopy(
                        denseVector.values, 0, values, currentOffset, denseVector.values.length);

                currentOffset += denseVector.size();
            }
        }
        return Vectors.dense(values);
    }

    /** Assembles the input columns into a sparse vector. */
    private static Vector assembleSparse(
            String[] inputCols, Row inputRow, int vectorSize, int nnz) {
        int[] indices = new int[nnz];
        double[] values = new double[nnz];

        int currentIndex = 0;
        int currentOffset = 0;

        for (String inputCol : inputCols) {
            Object object = inputRow.getField(inputCol);
            if (object instanceof Number) {
                indices[currentOffset] = currentIndex;
                values[currentOffset] = ((Number) object).doubleValue();
                currentOffset++;
                currentIndex++;
            } else if (object instanceof SparseVector) {
                SparseVector sparseVector = (SparseVector) object;
                for (int i = 0; i < sparseVector.indices.length; i++) {
                    indices[currentOffset + i] = sparseVector.indices[i] + currentIndex;
                }
                System.arraycopy(
                        sparseVector.values, 0, values, currentOffset, sparseVector.values.length);
                currentIndex += sparseVector.size();
                currentOffset += sparseVector.indices.length;
            } else {
                DenseVector denseVector = (DenseVector) object;
                for (int i = 0; i < denseVector.size(); ++i) {
                    indices[currentOffset + i] = i + currentIndex;
                }
                System.arraycopy(
                        denseVector.values, 0, values, currentOffset, denseVector.values.length);
                currentIndex += denseVector.size();
                currentOffset += denseVector.size();
            }
        }
        return new SparseVector(vectorSize, indices, values);
    }
}
