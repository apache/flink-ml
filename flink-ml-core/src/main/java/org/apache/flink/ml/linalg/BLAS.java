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

package org.apache.flink.ml.linalg;

import org.apache.flink.util.Preconditions;

/** A utility class that provides BLAS routines over matrices and vectors. */
public class BLAS {
    /** For level-1 function dspmv, use javaBLAS for better performance. */
    private static final dev.ludovic.netlib.BLAS JAVA_BLAS =
            dev.ludovic.netlib.JavaBLAS.getInstance();

    /** \sum_i |x_i| . */
    public static double asum(DenseVector x) {
        return JAVA_BLAS.dasum(x.size(), x.values, 0, 1);
    }

    /** y += a * x . */
    public static void axpy(double a, DenseVector x, DenseVector y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        JAVA_BLAS.daxpy(x.size(), a, x.values, 1, y.values, 1);
    }

    /** x \cdot y . */
    public static double dot(DenseVector x, DenseVector y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        return JAVA_BLAS.ddot(x.size(), x.values, 1, y.values, 1);
    }

    /** \sqrt(\sum_i x_i * x_i) . */
    public static double norm2(DenseVector x) {
        return JAVA_BLAS.dnrm2(x.size(), x.values, 1);
    }

    /** x = x * a . */
    public static void scal(double a, DenseVector x) {
        JAVA_BLAS.dscal(x.size(), a, x.values, 1);
    }

    /**
     * y = alpha * matrix * x + beta * y or y = alpha * (matrix^T) * x + beta * y.
     *
     * @param alpha The alpha value.
     * @param matrix Dense matrix with size m x n.
     * @param transMatrix Whether transposes matrix before multiply.
     * @param x Dense vector with size n.
     * @param beta The beta value.
     * @param y Dense vector with size m.
     */
    public static void gemv(
            double alpha,
            DenseMatrix matrix,
            boolean transMatrix,
            DenseVector x,
            double beta,
            DenseVector y) {
        Preconditions.checkArgument(
                transMatrix
                        ? (matrix.numRows() == x.size() && matrix.numCols() == y.size())
                        : (matrix.numRows() == y.size() && matrix.numCols() == x.size()),
                "Matrix and vector size mismatched.");
        final String trans = transMatrix ? "T" : "N";
        JAVA_BLAS.dgemv(
                trans,
                matrix.numRows(),
                matrix.numCols(),
                alpha,
                matrix.values,
                matrix.numRows(),
                x.values,
                1,
                beta,
                y.values,
                1);
    }
}
