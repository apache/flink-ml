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

package org.apache.flink.ml.feature.polynomialexpansion;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.TableUtils;
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
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.ArithmeticUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * A Transformer that expands the input vectors in polynomial space.
 *
 * <p>Take a 2-dimension vector as an example: `(x, y)`, if we want to expand it with degree 2, then
 * we get `(x, x * x, y, x * y, y * y)`.
 *
 * <p>For more information about the polynomial expansion, see
 * http://en.wikipedia.org/wiki/Polynomial_expansion.
 */
public class PolynomialExpansion
        implements Transformer<PolynomialExpansion>,
                PolynomialExpansionParams<PolynomialExpansion> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public PolynomialExpansion() {
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
                        .map(
                                new PolynomialExpansionFunction(getDegree(), getInputCol()),
                                outputTypeInfo);

        Table outputTable = tEnv.fromDataStream(output);
        return new Table[] {outputTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static PolynomialExpansion load(StreamTableEnvironment env, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Polynomial expansion function that expands a vector in polynomial space. This expansion is
     * done using recursion. Given input vector and degree, the size after expansion is (vectorSize
     * + degree) (including 1 and first-order values). For example, let f([a, b, c], 3) be the
     * function that expands [a, b, c] to their monomials of degree 3. We have the following
     * recursion:
     *
     * <blockquote>
     *
     * $$ f([a, b, c], 3) &= f([a, b], 3) ++ f([a, b], 2) * c ++ f([a, b], 1) * c^2 ++ [c^3] $$
     *
     * </blockquote>
     *
     * <p>To handle sparsity, if c is zero, we can skip all monomials that contain it. We remember
     * the current index and increment it properly for sparse input.
     */
    private static class PolynomialExpansionFunction implements MapFunction<Row, Row> {
        private final int degree;
        private final String inputCol;

        public PolynomialExpansionFunction(int degree, String inputCol) {
            this.degree = degree;
            this.inputCol = inputCol;
        }

        @Override
        public Row map(Row row) throws Exception {
            Vector<Integer, Double, int[], double[]> vec = row.getFieldAs(inputCol);
            if (vec == null) {
                throw new IllegalArgumentException("The vector must not be null.");
            }
            Vector outputVec;
            if (vec instanceof DenseVector) {
                int size = (int) vec.size();
                double[] retVals = new double[getResultVectorSize(size, degree) - 1];
                expandDenseVector(
                        ((DenseVector<Integer, Double, int[], double[]>) vec).getValues(),
                        size - 1,
                        degree,
                        1.0,
                        retVals,
                        -1);
                outputVec = Vectors.dense(retVals);
            } else if (vec instanceof SparseVector) {
                SparseVector<Integer, Double, int[], double[]> sparseVec =
                        (SparseVector<Integer, Double, int[], double[]>) vec;
                int[] indices = sparseVec.getIndices();
                double[] values = sparseVec.getValues();
                int size = (int) sparseVec.size();
                int nnz = sparseVec.getValues().length;
                int nnzPolySize = getResultVectorSize(nnz, degree);

                Tuple2<Integer, int[]> polyIndices = Tuple2.of(0, new int[nnzPolySize - 1]);
                Tuple2<Integer, double[]> polyValues = Tuple2.of(0, new double[nnzPolySize - 1]);
                expandSparseVector(
                        indices,
                        values,
                        nnz - 1,
                        size - 1,
                        degree,
                        1.0,
                        polyIndices,
                        polyValues,
                        -1);

                outputVec =
                        Vectors.sparse(
                                getResultVectorSize(size, degree) - 1,
                                polyIndices.f1,
                                polyValues.f1);
            } else {
                throw new UnsupportedOperationException(
                        "Only supports DenseVector or SparseVector.");
            }
            return Row.join(row, Row.of(outputVec));
        }

        /** Calculates the length of the expended vector. */
        private static int getResultVectorSize(int num, int degree) {
            if (num == 0) {
                return 1;
            }

            if (num == 1 || degree == 1) {
                return num + degree;
            }

            if (degree > num) {
                return getResultVectorSize(degree, num);
            }

            long res = 1;
            int i = num + 1;
            int j;

            if (num + degree < 61) {
                for (j = 1; j <= degree; ++j) {
                    res = res * i / j;
                    ++i;
                }
            } else {
                int depth;
                for (j = 1; j <= degree; ++j) {
                    depth = ArithmeticUtils.gcd(i, j);
                    res = ArithmeticUtils.mulAndCheck(res / (j / depth), i / depth);
                    ++i;
                }
            }

            if (res > Integer.MAX_VALUE) {
                throw new RuntimeException("The expended polynomial size is too large.");
            }
            return (int) res;
        }

        /** Expands the dense vector in polynomial space. */
        private static int expandDenseVector(
                double[] values,
                int lastIdx,
                int degree,
                double factor,
                double[] retValues,
                int curPolyIdx) {
            if (!Double.valueOf(factor).equals(0.0)) {
                if (degree == 0 || lastIdx < 0) {
                    if (curPolyIdx >= 0) {
                        retValues[curPolyIdx] = factor;
                    }
                } else {
                    double v = values[lastIdx];
                    int newLastIdx = lastIdx - 1;
                    double alpha = factor;
                    int i = 0;
                    int curStart = curPolyIdx;

                    while (i <= degree && Math.abs(alpha) > 0.0) {
                        curStart =
                                expandDenseVector(
                                        values, newLastIdx, degree - i, alpha, retValues, curStart);
                        i += 1;
                        alpha *= v;
                    }
                }
            }
            return curPolyIdx + getResultVectorSize(lastIdx + 1, degree);
        }

        /** Expands the sparse vector in polynomial space. */
        private static int expandSparseVector(
                int[] indices,
                double[] values,
                int lastIdx,
                int lastFeatureIdx,
                int degree,
                double factor,
                Tuple2<Integer, int[]> polyIndices,
                Tuple2<Integer, double[]> polyValues,
                int curPolyIdx) {
            if (!Double.valueOf(factor).equals(0.0)) {
                if (degree == 0 || lastIdx < 0) {
                    if (curPolyIdx >= 0) {
                        polyIndices.f1[polyIndices.f0] = curPolyIdx;
                        polyValues.f1[polyValues.f0] = factor;
                        polyIndices.f0++;
                        polyValues.f0++;
                    }
                } else {
                    double v = values[lastIdx];
                    int lastIdx1 = lastIdx - 1;
                    int lastFeatureIdx1 = indices[lastIdx] - 1;
                    double alpha = factor;
                    int curStart = curPolyIdx;
                    int i = 0;

                    while (i <= degree && Math.abs(alpha) > 0.0) {
                        curStart =
                                expandSparseVector(
                                        indices,
                                        values,
                                        lastIdx1,
                                        lastFeatureIdx1,
                                        degree - i,
                                        alpha,
                                        polyIndices,
                                        polyValues,
                                        curStart);
                        i++;
                        alpha *= v;
                    }
                }
            }
            return curPolyIdx + getResultVectorSize(lastFeatureIdx + 1, degree);
        }
    }
}
