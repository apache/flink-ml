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

package org.apache.flink.ml.feature.interaction;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
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
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * A Transformer that takes vector or numerical columns, and generates a single vector column that
 * contains the product of all combinations of one value from each input column.
 *
 * <p>For example, when the input feature values are Double(2) and Vector(3, 4), the output would be
 * Vector(6, 8). When the input feature values are Vector(1, 2) and Vector(3, 4), the output would
 * be Vector(3, 4, 6, 8). If you change the position of these two input Vectors, the output would be
 * Vector(3, 6, 4, 8).
 */
public class Interaction implements Transformer<Interaction>, InteractionParams<Interaction> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Interaction() {
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
                        .map(new InteractionFunction(getInputCols()), outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);

        return new Table[] {outputTable};
    }

    private static class InteractionFunction implements MapFunction<Row, Row> {
        private final String[] inputCols;
        private final int[] featureSize;
        private final int[][] featureIndices;
        private final double[][] featureValues;

        public InteractionFunction(String[] inputCols) {
            this.inputCols = inputCols;
            this.featureSize = new int[inputCols.length];
            this.featureIndices = new int[inputCols.length][];
            this.featureValues = new double[inputCols.length][];
        }

        @Override
        public Row map(Row value) {
            int nnz = 1;
            boolean hasSparse = false;

            for (int i = 0; i < inputCols.length; ++i) {
                Object obj = value.getField(inputCols[i]);
                if (obj == null) {
                    return Row.join(value, Row.of((Object) null));
                }

                if (obj instanceof DenseIntDoubleVector) {
                    featureSize[i] = ((IntDoubleVector) obj).size();
                    if (featureIndices[i] == null || featureIndices[i].length != featureSize[i]) {
                        featureIndices[i] = new int[featureSize[i]];
                        for (int j = 0; j < featureSize[i]; ++j) {
                            featureIndices[i][j] = j;
                        }
                    }

                    featureValues[i] = ((DenseIntDoubleVector) obj).values;
                    nnz *= featureSize[i];
                } else if (obj instanceof SparseIntDoubleVector) {
                    featureSize[i] = ((IntDoubleVector) obj).size();
                    featureIndices[i] = ((SparseIntDoubleVector) obj).indices;
                    featureValues[i] = ((SparseIntDoubleVector) obj).values;
                    nnz *= ((SparseIntDoubleVector) obj).values.length;
                    hasSparse = true;
                } else {
                    featureSize[i] = 1;
                    featureIndices[i] = new int[] {0};
                    featureValues[i] = new double[] {Double.parseDouble(obj.toString())};
                }
            }

            IntDoubleVector ret;
            int featureIter = inputCols.length - 1;
            if (hasSparse) {
                int[] indices = new int[nnz];
                double[] values = new double[nnz];
                Arrays.fill(values, 1.0);
                int offset = 1;
                int size = 1;

                while (featureIter >= 0) {
                    int[] prevIndices = featureIndices[featureIter];
                    double[] prevValues = featureValues[featureIter];

                    for (int i = 0; i < nnz / offset; ++i) {
                        int idxOffset = i * offset;
                        int idx = i % prevValues.length;
                        for (int j = 0; j < offset; ++j) {
                            values[idxOffset + j] *= prevValues[idx];
                            indices[idxOffset + j] += prevIndices[idx] * size;
                        }
                    }

                    offset *= prevValues.length;
                    size *= featureSize[featureIter--];
                }
                ret = Vectors.sparse(size, indices, values);
            } else {
                double[] values = new double[nnz];
                Arrays.fill(values, 1.0);
                int idxOffset = 1;

                while (featureIter >= 0) {
                    double[] prevValues = featureValues[featureIter--];
                    for (int i = 0; i < nnz / idxOffset; ++i) {
                        int innerOffset = i * idxOffset;
                        int idx = i % prevValues.length;
                        for (int j = 0; j < idxOffset; ++j) {
                            values[innerOffset + j] *= prevValues[idx];
                        }
                    }
                    idxOffset *= prevValues.length;
                }
                ret = new DenseIntDoubleVector(values);
            }

            return Row.join(value, Row.of(ret));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Interaction load(StreamTableEnvironment env, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
