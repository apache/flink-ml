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

package org.apache.flink.ml.feature.binarizer;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
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
 * A Transformer that binarizes the columns of continuous features by the given thresholds. The
 * continuous features may be DenseVector, SparseVector, or Numerical Value.
 */
public class Binarizer implements Transformer<Binarizer>, BinarizerParams<Binarizer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Binarizer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        String[] inputCols = getInputCols();
        Preconditions.checkArgument(inputCols.length == getThresholds().length);
        TypeInformation<?>[] outputTypes = new TypeInformation[inputCols.length];

        for (int i = 0; i < inputCols.length; ++i) {
            int idx = inputTypeInfo.getFieldIndex(inputCols[i]);
            Class<?> typeClass = inputTypeInfo.getTypeAt(idx).getTypeClass();
            if (typeClass.equals(SparseVector.class)) {
                outputTypes[i] = SparseVectorTypeInfo.INSTANCE;
            } else if (typeClass.equals(DenseVector.class)) {
                outputTypes[i] = DenseVectorTypeInfo.INSTANCE;
            } else if (typeClass.equals(Vector.class)) {
                outputTypes[i] = VectorTypeInfo.INSTANCE;
            } else {
                outputTypes[i] = Types.DOUBLE;
            }
        }

        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), outputTypes),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCols()));

        DataStream<Row> output =
                tEnv.toDataStream(inputs[0])
                        .map(new BinarizeFunction(inputCols, getThresholds()), outputTypeInfo);
        Table outputTable = tEnv.fromDataStream(output);

        return new Table[] {outputTable};
    }

    private static class BinarizeFunction implements MapFunction<Row, Row> {
        private final String[] inputCols;
        private final Double[] thresholds;

        public BinarizeFunction(String[] inputCols, Double[] thresholds) {
            this.inputCols = inputCols;
            this.thresholds = thresholds;
        }

        @Override
        public Row map(Row input) {
            if (null == input) {
                return null;
            }

            Row result = new Row(inputCols.length);
            for (int i = 0; i < inputCols.length; ++i) {
                result.setField(i, binarizerFunc(input.getField(inputCols[i]), thresholds[i]));
            }
            return Row.join(input, result);
        }

        private Object binarizerFunc(Object obj, double threshold) {
            if (obj instanceof DenseVector) {
                DenseVector inputVec = (DenseVector) obj;
                DenseVector vec = inputVec.clone();
                for (int i = 0; i < vec.size(); ++i) {
                    vec.values[i] = inputVec.get(i) > threshold ? 1.0 : 0.0;
                }
                return vec;
            } else if (obj instanceof SparseVector) {
                SparseVector inputVec = (SparseVector) obj;
                int[] newIndices = new int[inputVec.indices.length];
                int pos = 0;

                for (int i = 0; i < inputVec.indices.length; ++i) {
                    if (inputVec.values[i] > threshold) {
                        newIndices[pos++] = inputVec.indices[i];
                    }
                }

                double[] newValues = new double[pos];
                Arrays.fill(newValues, 1.0);
                return new SparseVector(inputVec.size(), Arrays.copyOf(newIndices, pos), newValues);
            } else {
                return Double.parseDouble(obj.toString()) > threshold ? 1.0 : 0.0;
            }
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Binarizer load(StreamTableEnvironment env, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
