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

package org.apache.flink.ml.param;

import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.SparseLongDoubleVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.JsonUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Class for the Vector parameter. */
public class VectorParam extends Param<Vector> {
    private static final String INDICES_NAME = "indices";
    private static final String VALUES_NAME = "values";

    public VectorParam(
            String name,
            String description,
            Vector defaultValue,
            ParamValidator<Vector> validator) {
        super(name, Vector.class, description, defaultValue, validator);
    }

    public VectorParam(String name, String description, Vector defaultValue) {
        this(name, description, defaultValue, ParamValidators.alwaysTrue());
    }

    @Override
    public Object jsonEncode(Vector value) throws IOException {
        Map<String, Object> map = new HashMap<>();
        map.put("class", value.getClass().getName());
        if (value instanceof SparseIntDoubleVector) {
            SparseIntDoubleVector vector = (SparseIntDoubleVector) value;
            map.put("n", vector.size());
            map.put(INDICES_NAME, JsonUtils.OBJECT_MAPPER.writeValueAsString(vector.getIndices()));
            map.put(VALUES_NAME, JsonUtils.OBJECT_MAPPER.writeValueAsString(vector.getValues()));
        } else if (value instanceof DenseIntDoubleVector) {
            DenseIntDoubleVector vector = (DenseIntDoubleVector) value;
            map.put("n", vector.size());
            map.put(VALUES_NAME, JsonUtils.OBJECT_MAPPER.writeValueAsString(vector.getValues()));
        } else if (value instanceof SparseLongDoubleVector) {
            SparseLongDoubleVector vector = (SparseLongDoubleVector) value;
            map.put("n", vector.size());
            map.put(INDICES_NAME, JsonUtils.OBJECT_MAPPER.writeValueAsString(vector.getIndices()));
            map.put(VALUES_NAME, JsonUtils.OBJECT_MAPPER.writeValueAsString(vector.getValues()));
        } else {
            throw new UnsupportedOperationException("Vector parameter is invalid.");
        }
        return map;
    }

    @Override
    public Vector jsonDecode(Object object) throws IOException {
        Map<String, Object> map = (Map) object;

        String classString = (String) map.get("class");
        if (classString == null) {
            // For compatibility with the decode method before.
            // TODO: Deprecate this branch.
            if (map.size() == 1) {
                List<Double> list = (List<Double>) map.get("values");
                double[] values = new double[list.size()];
                for (int i = 0; i < values.length; ++i) {
                    values[i] = list.get(i);
                }
                return Vectors.dense(values);
            } else if (map.size() == 3) {
                List<Double> valuesList = (List<Double>) map.get("values");
                List<Integer> indicesList = (List<Integer>) map.get("indices");
                int n = (int) map.get("n");
                double[] values = new double[valuesList.size()];
                int[] indices = new int[indicesList.size()];
                for (int i = 0; i < values.length; ++i) {
                    values[i] = valuesList.get(i);
                    indices[i] = indicesList.get(i);
                }
                return Vectors.sparse(n, indices, values);
            } else {
                throw new UnsupportedOperationException("Vector parameter is invalid.");
            }
        } else {
            if (classString.equals(SparseIntDoubleVector.class.getName())) {
                long n = ((Number) map.get("n")).longValue();
                int[] indices =
                        JsonUtils.OBJECT_MAPPER.readValue(
                                (String) map.get(INDICES_NAME), int[].class);
                double[] values =
                        JsonUtils.OBJECT_MAPPER.readValue(
                                (String) map.get(VALUES_NAME), double[].class);
                return Vectors.sparse(n, indices, values);
            } else if (classString.equals(DenseIntDoubleVector.class.getName())) {
                double[] values =
                        JsonUtils.OBJECT_MAPPER.readValue(
                                (String) map.get(VALUES_NAME), double[].class);
                return Vectors.dense(values);
            } else if (classString.equals(SparseLongDoubleVector.class.getName())) {
                long n = ((Number) map.get("n")).longValue();
                long[] indices =
                        JsonUtils.OBJECT_MAPPER.readValue(
                                (String) map.get(INDICES_NAME), long[].class);
                double[] values =
                        JsonUtils.OBJECT_MAPPER.readValue(
                                (String) map.get(VALUES_NAME), double[].class);
                return Vectors.sparse(n, indices, values);
            } else {
                throw new UnsupportedOperationException("Vector parameter is invalid.");
            }
        }
    }
}
