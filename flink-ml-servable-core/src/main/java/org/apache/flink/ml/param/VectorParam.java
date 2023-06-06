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
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;

import java.util.List;
import java.util.Map;

/** Class for the Vector parameter. */
public class VectorParam extends Param<IntDoubleVector> {

    public VectorParam(
            String name,
            String description,
            IntDoubleVector defaultValue,
            ParamValidator<IntDoubleVector> validator) {
        super(name, IntDoubleVector.class, description, defaultValue, validator);
    }

    public VectorParam(String name, String description, IntDoubleVector defaultValue) {
        this(name, description, defaultValue, ParamValidators.alwaysTrue());
    }

    @Override
    public IntDoubleVector jsonDecode(Object object) {
        Map<String, Object> vecValues = (Map) object;
        if (vecValues.size() == 1) {
            List<Double> list = (List<Double>) vecValues.get("values");
            double[] values = new double[list.size()];
            for (int i = 0; i < values.length; ++i) {
                values[i] = list.get(i);
            }
            return new DenseIntDoubleVector(values);
        } else if (vecValues.size() == 3) {
            List<Double> valuesList = (List<Double>) vecValues.get("values");
            List<Integer> indicesList = (List<Integer>) vecValues.get("indices");
            int n = (int) vecValues.get("n");
            double[] values = new double[valuesList.size()];
            int[] indices = new int[indicesList.size()];
            for (int i = 0; i < values.length; ++i) {
                values[i] = valuesList.get(i);
                indices[i] = indicesList.get(i);
            }
            return new SparseIntDoubleVector(n, indices, values);
        } else {
            throw new UnsupportedOperationException("Vector parameter is invalid.");
        }
    }
}
