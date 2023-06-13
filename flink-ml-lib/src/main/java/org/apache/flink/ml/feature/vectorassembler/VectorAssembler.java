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
import org.apache.flink.api.java.tuple.Tuple2;
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
 * A Transformer which combines a given list of input columns into a vector column. Input columns
 * would be numerical or vectors whose sizes are specified by the {@link #INPUT_SIZES} parameter.
 * Invalid input data with null values or values with wrong sizes would be dealt with according to
 * the strategy specified by the {@link HasHandleInvalid} parameter as follows:
 *
 * <ul>
 *   <li>keep: If the input column data is null, a vector would be created with the specified size
 *       and NaN values. The vector would be used in the assembling process to represent the input
 *       column data. If the input column data is a vector, the data would be used in the assembling
 *       process even if it has a wrong size.
 *   <li>skip: If the input column data is null or a vector with wrong size, the input row would be
 *       filtered out and not be sent to downstream operators.
 *   <li>error: If the input column data is null or a vector with wrong size, an exception would be
 *       thrown.
 * </ul>
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
        Preconditions.checkArgument(getInputSizes().length == getInputCols().length);
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
                                new AssemblerFunction(
                                        getInputCols(), getHandleInvalid(), getInputSizes()),
                                outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);
        return new Table[] {outputTable};
    }

    private static class AssemblerFunction implements FlatMapFunction<Row, Row> {
        private final String[] inputCols;
        private final String handleInvalid;
        private final Integer[] inputSizes;
        private final boolean keepInvalid;

        public AssemblerFunction(String[] inputCols, String handleInvalid, Integer[] inputSizes) {
            this.inputCols = inputCols;
            this.handleInvalid = handleInvalid;
            this.inputSizes = inputSizes;
            keepInvalid = handleInvalid.equals(HasHandleInvalid.KEEP_INVALID);
        }

        @Override
        public void flatMap(Row value, Collector<Row> out) {
            try {
                Tuple2<Integer, Integer> vectorSizeAndNnz = computeVectorSizeAndNnz(value);
                int vectorSize = vectorSizeAndNnz.f0;
                int nnz = vectorSizeAndNnz.f1;
                Vector assembledVec =
                        nnz * RATIO > vectorSize
                                ? assembleDense(inputCols, value, vectorSize)
                                : assembleSparse(inputCols, value, vectorSize, nnz);
                out.collect(Row.join(value, Row.of(assembledVec)));
            } catch (Exception e) {
                if (handleInvalid.equals(ERROR_INVALID)) {
                    throw new RuntimeException("Vector assembler failed with exception : " + e);
                }
            }
        }

        private Tuple2<Integer, Integer> computeVectorSizeAndNnz(Row value) {
            int vectorSize = 0;
            int nnz = 0;
            for (int i = 0; i < inputCols.length; ++i) {
                Object object = value.getField(inputCols[i]);
                if (object != null) {
                    if (object instanceof Number) {
                        checkSize(inputSizes[i], 1);
                        if (Double.isNaN(((Number) object).doubleValue()) && !keepInvalid) {
                            throw new RuntimeException(
                                    "Encountered NaN while assembling a row with handleInvalid = 'error'. Consider "
                                            + "removing NaNs from dataset or using handleInvalid = 'keep' or 'skip'.");
                        }
                        vectorSize += 1;
                        nnz += 1;
                    } else if (object instanceof SparseVector) {
                        int localSize = (int) ((SparseVector) object).size();
                        checkSize(inputSizes[i], localSize);
                        nnz += ((SparseVector) object).indices.length;
                        vectorSize += localSize;
                    } else if (object instanceof DenseVector) {
                        int localSize = (int) ((DenseVector) object).size();
                        checkSize(inputSizes[i], localSize);
                        vectorSize += localSize;
                        nnz += ((DenseVector) object).size();
                    } else {
                        throw new IllegalArgumentException(
                                String.format(
                                        "Input type %s has not been supported yet. Only Vector and Number types are supported.",
                                        object.getClass()));
                    }
                } else {
                    vectorSize += inputSizes[i];
                    nnz += inputSizes[i];
                    if (keepInvalid) {
                        if (inputSizes[i] > 1) {
                            DenseVector tmpVec = new DenseVector(inputSizes[i]);
                            for (int j = 0; j < inputSizes[i]; ++j) {
                                tmpVec.values[j] = Double.NaN;
                            }
                            value.setField(inputCols[i], tmpVec);
                        } else {
                            value.setField(inputCols[i], Double.NaN);
                        }
                    } else {
                        throw new RuntimeException(
                                "Input column value is null. Please check the input data or using handleInvalid = 'keep'.");
                    }
                }
            }
            return Tuple2.of(vectorSize, nnz);
        }

        private void checkSize(int expectedSize, int currentSize) {
            if (keepInvalid) {
                return;
            }
            if (currentSize != expectedSize) {
                throw new IllegalArgumentException(
                        String.format(
                                "Input vector/number size does not meet with expected. Expected size: %d, actual size: %s.",
                                expectedSize, currentSize));
            }
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
                        denseVector.values, 0, values, currentOffset, (int) denseVector.size());

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
