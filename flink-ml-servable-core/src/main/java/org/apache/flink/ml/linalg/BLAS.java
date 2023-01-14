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
    public static void axpy(double a, Vector x, DenseVector y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        axpy(a, x, y, x.size());
    }

    /** y += a * x for the first k dimensions, with the other dimensions unchanged. */
    public static void axpy(double a, Vector x, DenseVector y, int k) {
        Preconditions.checkArgument(x.size() >= k && y.size() >= k);
        if (x instanceof SparseVector) {
            axpy(a, (SparseVector) x, y, k);
        } else {
            axpy(a, (DenseVector) x, y, k);
        }
    }

    /** Computes the hadamard product of the two vectors (y = y \hdot x). */
    public static void hDot(Vector x, Vector y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        if (x instanceof SparseVector) {
            if (y instanceof SparseVector) {
                hDot((SparseVector) x, (SparseVector) y);
            } else {
                hDot((SparseVector) x, (DenseVector) y);
            }
        } else {
            if (y instanceof SparseVector) {
                hDot((DenseVector) x, (SparseVector) y);
            } else {
                hDot((DenseVector) x, (DenseVector) y);
            }
        }
    }

    /** Computes the dot of the two vectors (y \dot x). */
    public static double dot(Vector x, Vector y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        if (x instanceof SparseVector) {
            if (y instanceof SparseVector) {
                return dot((SparseVector) x, (SparseVector) y);
            } else {
                return dot((DenseVector) y, (SparseVector) x);
            }
        } else {
            if (y instanceof SparseVector) {
                return dot((DenseVector) x, (SparseVector) y);
            } else {
                return dot((DenseVector) x, (DenseVector) y);
            }
        }
    }

    private static double dot(DenseVector x, DenseVector y) {
        return JAVA_BLAS.ddot(x.size(), x.values, 1, y.values, 1);
    }

    private static double dot(DenseVector x, SparseVector y) {
        double dotValue = 0.0;
        for (int i = 0; i < y.indices.length; ++i) {
            dotValue += y.values[i] * x.values[y.indices[i]];
        }
        return dotValue;
    }

    private static double dot(SparseVector x, SparseVector y) {
        double dotValue = 0;
        int p0 = 0;
        int p1 = 0;
        while (p0 < x.values.length && p1 < y.values.length) {
            if (x.indices[p0] == y.indices[p1]) {
                dotValue += x.values[p0] * y.values[p1];
                p0++;
                p1++;
            } else if (x.indices[p0] < y.indices[p1]) {
                p0++;
            } else {
                p1++;
            }
        }
        return dotValue;
    }

    /** \sqrt(\sum_i x_i * x_i) . */
    public static double norm2(Vector x) {
        if (x instanceof DenseVector) {
            return norm2((DenseVector) x);
        }
        return norm2((SparseVector) x);
    }

    private static double norm2(DenseVector x) {
        return JAVA_BLAS.dnrm2(x.size(), x.values, 1);
    }

    private static double norm2(SparseVector x) {
        return JAVA_BLAS.dnrm2(x.values.length, x.values, 1);
    }

    /** Calculates the p-norm of the vector x. */
    public static double norm(Vector x, double p) {
        Preconditions.checkArgument(p >= 1.0, "p value must >= 1.0, but the current p is : " + p);
        double norm = 0.0;
        double[] data =
                (x instanceof DenseVector) ? ((DenseVector) x).values : ((SparseVector) x).values;

        if (p == 1.0) {
            for (double datum : data) {
                norm += Math.abs(datum);
            }
        } else if (p == 2.0) {
            norm = norm2(x);
        } else if (p == Double.POSITIVE_INFINITY) {
            for (double datum : data) {
                norm = Math.max(Math.abs(datum), norm);
            }
        } else {
            for (double datum : data) {
                norm += Math.pow(Math.abs(datum), p);
            }
            norm = Math.pow(norm, 1.0 / p);
        }

        return norm;
    }

    /** x = x * a . */
    public static void scal(double a, Vector x) {
        if (x instanceof DenseVector) {
            JAVA_BLAS.dscal(x.size(), a, ((DenseVector) x).values, 1);
        } else {
            double[] values = ((SparseVector) x).values;
            JAVA_BLAS.dscal(values.length, a, values, 1);
        }
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

    private static void axpy(double a, DenseVector x, DenseVector y, int k) {
        JAVA_BLAS.daxpy(k, a, x.values, 1, y.values, 1);
    }

    private static void axpy(double a, SparseVector x, DenseVector y, int k) {
        for (int i = 0; i < x.indices.length; i++) {
            int index = x.indices[i];
            if (index >= k) {
                return;
            }
            y.values[index] += a * x.values[i];
        }
    }

    private static void hDot(SparseVector x, SparseVector y) {
        int idx = 0;
        int idy = 0;
        while (idx < x.indices.length && idy < y.indices.length) {
            int indexX = x.indices[idx];
            while (idy < y.indices.length && y.indices[idy] < indexX) {
                y.values[idy] = 0;
                idy++;
            }
            if (idy < y.indices.length && y.indices[idy] == indexX) {
                y.values[idy] *= x.values[idx];
                idy++;
            }
            idx++;
        }
        while (idy < y.indices.length) {
            y.values[idy] = 0;
            idy++;
        }
    }

    private static void hDot(SparseVector x, DenseVector y) {
        int idx = 0;
        for (int i = 0; i < y.size(); i++) {
            if (idx < x.indices.length && x.indices[idx] == i) {
                y.values[i] *= x.values[idx];
                idx++;
            } else {
                y.values[i] = 0;
            }
        }
    }

    private static void hDot(DenseVector x, SparseVector y) {
        for (int i = 0; i < y.values.length; i++) {
            y.values[i] *= x.values[y.indices[i]];
        }
    }

    private static void hDot(DenseVector x, DenseVector y) {
        for (int i = 0; i < x.values.length; i++) {
            y.values[i] *= x.values[i];
        }
    }
}
