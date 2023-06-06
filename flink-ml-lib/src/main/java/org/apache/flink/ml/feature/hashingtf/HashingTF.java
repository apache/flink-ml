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

package org.apache.flink.ml.feature.hashingtf;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.SparseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.apache.flink.shaded.guava30.com.google.common.hash.Hashing.murmur3_32;

/**
 * A Transformer that maps a sequence of terms(strings, numbers, booleans) to a sparse vector with a
 * specified dimension using the hashing trick.
 *
 * <p>If multiple features are projected into the same column, the output values are accumulated by
 * default. Users could also enforce all non-zero output values as 1 by setting {@link
 * HashingTFParams#BINARY} as true.
 *
 * <p>For the hashing trick, see https://en.wikipedia.org/wiki/Feature_hashing for details.
 */
public class HashingTF implements Transformer<HashingTF>, HashingTFParams<HashingTF> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    private static final org.apache.flink.shaded.guava30.com.google.common.hash.HashFunction
            HASH_FUNC = murmur3_32(0);

    public HashingTF() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        ResolvedSchema tableSchema = inputs[0].getResolvedSchema();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(tableSchema);
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                SparseIntDoubleVectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol()));

        DataStream<Row> output =
                tEnv.toDataStream(inputs[0])
                        .map(
                                new HashTFFunction(getInputCol(), getBinary(), getNumFeatures()),
                                outputTypeInfo);
        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    public static HashingTF load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    /** The main logic of {@link HashingTF}, which converts the input to a sparse vector. */
    public static class HashTFFunction implements MapFunction<Row, Row> {
        private final String inputCol;
        private final boolean binary;
        private final int numFeatures;

        public HashTFFunction(String inputCol, boolean binary, int numFeatures) {
            this.inputCol = inputCol;
            this.binary = binary;
            this.numFeatures = numFeatures;
        }

        @Override
        public Row map(Row row) throws Exception {
            Object inputObj = row.getField(inputCol);

            Iterable<Object> inputList;
            if (inputObj.getClass().isArray()) {
                inputList = Arrays.asList((Object[]) inputObj);
            } else if (inputObj instanceof Iterable) {
                inputList = (Iterable<Object>) inputObj;
            } else {
                throw new IllegalArgumentException(
                        "Input format "
                                + inputObj.getClass().getCanonicalName()
                                + " is not supported for input column "
                                + inputCol
                                + ". Supported options are Array and Iterable.");
            }

            Map<Integer, Integer> map = new HashMap<>();
            for (Object obj : inputList) {
                int hashValue = hash(obj);
                int index = nonNegativeMod(hashValue, numFeatures);
                if (map.containsKey(index)) {
                    if (!binary) {
                        map.put(index, map.get(index) + 1);
                    }
                } else {
                    map.put(index, 1);
                }
            }

            // Converts from map to a sparse vector.
            int[] indices = new int[map.size()];
            double[] values = new double[map.size()];
            int idx = 0;
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                indices[idx] = entry.getKey();
                values[idx] = entry.getValue();
                idx++;
            }
            return Row.join(row, Row.of(Vectors.sparse(numFeatures, indices, values)));
        }
    }

    private static int hash(Object obj) {
        if (obj == null) {
            return 0;
        } else if (obj instanceof Boolean) {
            int value = (Boolean) obj ? 1 : 0;
            return HASH_FUNC.hashInt(value).asInt();
        } else if (obj instanceof Byte) {
            byte value = (Byte) obj;
            return HASH_FUNC.hashInt(value).asInt();
        } else if (obj instanceof Short) {
            short value = (Short) obj;
            return HASH_FUNC.hashInt(value).asInt();
        } else if (obj instanceof Integer) {
            int value = (Integer) obj;
            return HASH_FUNC.hashInt(value).asInt();
        } else if (obj instanceof Long) {
            long value = (Long) obj;
            return HASH_FUNC.hashLong(value).asInt();
        } else if (obj instanceof Float) {
            float value = (Float) obj;
            return HASH_FUNC.hashInt(Float.floatToIntBits(value)).asInt();
        } else if (obj instanceof Double) {
            double value = (Double) obj;
            return HASH_FUNC.hashLong(Double.doubleToLongBits(value)).asInt();
        } else if (obj instanceof String) {
            return HASH_FUNC.hashUnencodedChars((String) obj).asInt();
        } else {
            throw new UnsupportedOperationException(
                    "HashingTF does not support type "
                            + obj.getClass().getCanonicalName()
                            + " of input data.");
        }
    }

    private static int nonNegativeMod(int x, int mod) {
        int rawMod = x % mod;
        return rawMod < 0 ? rawMod + mod : rawMod;
    }
}
