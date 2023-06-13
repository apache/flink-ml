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

package org.apache.flink.ml;

import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.table.api.ApiExpression;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.catalog.DataTypeFactory;
import org.apache.flink.table.functions.ScalarFunction;
import org.apache.flink.table.types.inference.TypeInference;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Optional;

import static org.apache.flink.table.api.Expressions.call;

/** Built-in table functions for data transformations. */
@SuppressWarnings("unused")
public class Functions {
    /** Converts a column of {@link Vector}s into a column of double arrays. */
    public static ApiExpression vectorToArray(Object... arguments) {
        return call(VectorToArrayFunction.class, arguments);
    }

    /**
     * A {@link ScalarFunction} that converts a column of {@link Vector}s into a column of double
     * arrays.
     */
    public static class VectorToArrayFunction extends ScalarFunction {
        public double[] eval(Vector<Integer, Double, int[], double[]> vector) {
            return vector.toArray();
        }

        @Override
        public TypeInference getTypeInference(DataTypeFactory typeFactory) {
            return TypeInference.newBuilder()
                    .outputTypeStrategy(
                            callContext ->
                                    Optional.of(
                                            DataTypes.ARRAY(
                                                    DataTypes.DOUBLE()
                                                            .notNull()
                                                            .bridgedTo(double.class))))
                    .build();
        }
    }

    /**
     * Converts a column of arrays of numeric type into a column of {@link DenseVector} instances.
     */
    public static ApiExpression arrayToVector(Object... arguments) {
        return call(ArrayToVectorFunction.class, arguments);
    }

    /**
     * A {@link ScalarFunction} that converts a column of arrays of numeric type into a column of
     * {@link DenseVector} instances.
     */
    public static class ArrayToVectorFunction extends ScalarFunction {
        public DenseVector eval(double[] array) {
            return Vectors.dense(array);
        }

        public DenseVector eval(Double[] array) {
            return eval(ArrayUtils.toPrimitive(array));
        }

        public DenseVector eval(Number[] array) {
            double[] doubles = new double[array.length];
            for (int i = 0; i < array.length; i++) {
                doubles[i] = array[i].doubleValue();
            }
            return eval(doubles);
        }

        @Override
        public TypeInference getTypeInference(DataTypeFactory typeFactory) {
            return TypeInference.newBuilder()
                    .outputTypeStrategy(
                            callContext ->
                                    Optional.of(
                                            DataTypes.of(DenseVectorTypeInfo.INSTANCE)
                                                    .toDataType(typeFactory)))
                    .build();
        }
    }
}
