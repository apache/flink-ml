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
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A feature transformer that combines a given list of input columns into a vector column. Types of
 * input columns must be either vector or numerical value.
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
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(), TypeInformation.of(Vector.class)),
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
        public void flatMap(Row value, Collector<Row> out) throws Exception {
            try {
                Object[] objects = new Object[inputCols.length];
                for (int i = 0; i < objects.length; ++i) {
                    objects[i] = value.getField(inputCols[i]);
                }
                Vector assembledVector = assemble(objects);
                out.collect(Row.join(value, Row.of(assembledVector)));
            } catch (Exception e) {
                switch (handleInvalid) {
                    case VectorAssemblerParams.ERROR_INVALID:
                        throw e;
                    case VectorAssemblerParams.SKIP_INVALID:
                        return;
                    case VectorAssemblerParams.KEEP_INVALID:
                        out.collect(Row.join(value, Row.of((Object) null)));
                        return;
                    default:
                        throw new UnsupportedOperationException(
                                "handleInvalid=" + handleInvalid + " is not supported");
                }
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

    private static Vector assemble(Object[] objects) {
        int offset = 0;
        Map<Integer, Double> map = new LinkedHashMap<>(objects.length);
        for (Object object : objects) {
            Preconditions.checkNotNull(object, "Input column value should not be null.");
            if (object instanceof Number) {
                map.put(offset++, ((Number) object).doubleValue());
            } else if (object instanceof Vector) {
                offset = appendVector((Vector) object, map, offset);
            } else {
                throw new IllegalArgumentException("Input type has not been supported yet.");
            }
        }

        if (map.size() * RATIO > offset) {
            DenseVector assembledVector = new DenseVector(offset);
            for (int key : map.keySet()) {
                assembledVector.values[key] = map.get(key);
            }
            return assembledVector;
        } else {
            return convertMapToSparseVector(offset, map);
        }
    }

    private static int appendVector(Vector vec, Map<Integer, Double> map, int offset) {
        if (vec instanceof SparseVector) {
            SparseVector sparseVector = (SparseVector) vec;
            int[] indices = sparseVector.indices;
            double[] values = sparseVector.values;
            for (int i = 0; i < indices.length; ++i) {
                map.put(offset + indices[i], values[i]);
            }
            offset += sparseVector.size();
        } else {
            DenseVector denseVector = (DenseVector) vec;
            for (int i = 0; i < denseVector.size(); ++i) {
                map.put(offset++, denseVector.values[i]);
            }
        }
        return offset;
    }

    private static SparseVector convertMapToSparseVector(int size, Map<Integer, Double> map) {
        int[] indices = new int[map.size()];
        double[] values = new double[map.size()];
        int offset = 0;
        for (Map.Entry<Integer, Double> entry : map.entrySet()) {
            indices[offset] = entry.getKey();
            values[offset++] = entry.getValue();
        }
        return new SparseVector(size, indices, values);
    }
}
