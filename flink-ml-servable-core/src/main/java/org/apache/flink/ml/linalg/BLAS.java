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
    public static double asum(DenseVector<Integer, Double, int[], double[]> x) {
        return JAVA_BLAS.dasum((int) x.size(), x.getValues(), 0, 1);
    }

    /** y += a * x . */
    public static void axpy(
            double a,
            Vector<Integer, Double, int[], double[]> x,
            DenseVector<Integer, Double, int[], double[]> y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        axpy(a, x, y, (int) x.size());
    }

    /** y += a * x for the first k dimensions, with the other dimensions unchanged. */
    public static void axpy(
            double a,
            Vector<Integer, Double, int[], double[]> x,
            DenseVector<Integer, Double, int[], double[]> y,
            int k) {
        Preconditions.checkArgument(x.size() >= k && y.size() >= k);
        if (x instanceof SparseVector) {
            axpy(a, (SparseVector<Integer, Double, int[], double[]>) x, y, k);
        } else if (x instanceof DenseVector) {
            axpy(a, (DenseVector<Integer, Double, int[], double[]>) x, y, k);
        }
    }

    /** Computes the hadamard product of the two vectors (y = y \hdot x). */
    public static void hDot(
            Vector<Integer, Double, int[], double[]> x,
            Vector<Integer, Double, int[], double[]> y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        if (x instanceof SparseIntDoubleVector) {
            if (y instanceof SparseIntDoubleVector) {
                hDot((SparseIntDoubleVector) x, (SparseIntDoubleVector) y);
            } else {
                hDot((SparseIntDoubleVector) x, (DenseIntDoubleVector) y);
            }
        } else {
            if (y instanceof SparseIntDoubleVector) {
                hDot((DenseIntDoubleVector) x, (SparseIntDoubleVector) y);
            } else {
                hDot((DenseIntDoubleVector) x, (DenseIntDoubleVector) y);
            }
        }
    }

    /** Computes the dot of the two vectors (y \dot x). */
    public static double dot(
            Vector<Integer, Double, int[], double[]> x,
            Vector<Integer, Double, int[], double[]> y) {
        Preconditions.checkArgument(x.size() == y.size(), "Vector size mismatched.");
        if (x instanceof SparseVector) {
            if (y instanceof SparseVector) {
                return dot((SparseVector<Integer, Double, int[], double[]>) x, (SparseVector) y);
            } else {
                return dot((DenseIntDoubleVector) y, (SparseIntDoubleVector) x);
            }
        } else {
            if (y instanceof SparseIntDoubleVector) {
                return dot((DenseIntDoubleVector) x, (SparseIntDoubleVector) y);
            } else {
                return dot((DenseIntDoubleVector) x, (DenseIntDoubleVector) y);
            }
        }
    }

    private static double dot(
            DenseVector<Integer, Double, int[], double[]> x,
            DenseVector<Integer, Double, int[], double[]> y) {
        return JAVA_BLAS.ddot((int) x.size(), x.getValues(), 1, y.getValues(), 1);
    }

    private static double dot(
            DenseVector<Integer, Double, int[], double[]> x,
            SparseVector<Integer, Double, int[], double[]> y) {
        double dotValue = 0.0;
        int[] yi = y.getIndices();
        double[] yv = y.getValues();
        double[] xv = x.getValues();
        for (int i = 0; i < yi.length; ++i) {
            dotValue += yv[i] * xv[yi[i]];
        }
        return dotValue;
    }

    private static double dot(
            SparseVector<Integer, Double, int[], double[]> x,
            SparseVector<Integer, Double, int[], double[]> y) {
        double dotValue = 0;
        int p0 = 0;
        int p1 = 0;
        int[] xi = x.getIndices();
        int[] yi = y.getIndices();
        double[] xv = x.getValues();
        double[] yv = y.getValues();
        while (p0 < xv.length && p1 < yv.length) {
            if (xi[p0] == yi[p1]) {
                dotValue += xv[p0] * yv[p1];
                p0++;
                p1++;
            } else if (xi[p0] < yi[p1]) {
                p0++;
            } else {
                p1++;
            }
        }
        return dotValue;
    }

    /** \sqrt(\sum_i x_i * x_i) . */
    public static double norm2(Vector<Integer, Double, int[], double[]> x) {
        if (x instanceof DenseIntDoubleVector) {
            return norm2((DenseIntDoubleVector) x);
        }
        return norm2((SparseIntDoubleVector) x);
    }

    private static double norm2(DenseVector<Integer, Double, int[], double[]> x) {
        return JAVA_BLAS.dnrm2((int) x.size(), x.getValues(), 1);
    }

    private static double norm2(SparseVector<Integer, Double, int[], double[]> x) {
        return JAVA_BLAS.dnrm2(x.getValues().length, x.getValues(), 1);
    }

    /** Calculates the p-norm of the vector x. */
    public static double norm(Vector<Integer, Double, int[], double[]> x, double p) {
        Preconditions.checkArgument(p >= 1.0, "p value must >= 1.0, but the current p is : " + p);
        double norm = 0.0;
        double[] data =
                (x instanceof DenseIntDoubleVector)
                        ? ((DenseIntDoubleVector) x).getValues()
                        : ((SparseIntDoubleVector) x).getValues();

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
    public static void scal(double a, Vector<Integer, Double, int[], double[]> x) {
        if (x instanceof DenseIntDoubleVector) {
            JAVA_BLAS.dscal((int) x.size(), a, ((DenseIntDoubleVector) x).getValues(), 1);
        } else {
            double[] values = ((SparseIntDoubleVector) x).getValues();
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
            DenseVector<Integer, Double, int[], double[]> x,
            double beta,
            DenseVector<Integer, Double, int[], double[]> y) {
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
                x.getValues(),
                1,
                beta,
                y.getValues(),
                1);
    }

    private static void axpy(
            double a,
            DenseVector<Integer, Double, int[], double[]> x,
            DenseVector<Integer, Double, int[], double[]> y,
            int k) {
        JAVA_BLAS.daxpy(k, a, x.getValues(), 1, y.getValues(), 1);
    }

    private static void axpy(
            double a,
            SparseVector<Integer, Double, int[], double[]> x,
            DenseVector<Integer, Double, int[], double[]> y,
            int k) {
        int[] xi = x.getIndices();
        double[] xv = x.getValues();
        double[] yv = y.getValues();
        for (int i = 0; i < xi.length; i++) {
            int index = xi[i];
            if (index >= k) {
                return;
            }
            yv[index] += a * xv[i];
        }
    }

    private static void hDot(
            SparseVector<Integer, Double, int[], double[]> x,
            SparseVector<Integer, Double, int[], double[]> y) {
        int idx = 0;
        int idy = 0;
        int[] xi = x.getIndices();
        double[] xv = x.getValues();
        int[] yi = y.getIndices();
        double[] yv = y.getValues();

        while (idx < xi.length && idy < yi.length) {
            int indexX = xi[idx];
            while (idy < yi.length && yi[idy] < indexX) {
                yv[idy] = 0;
                idy++;
            }
            if (idy < yi.length && yi[idy] == indexX) {
                yv[idy] *= xv[idx];
                idy++;
            }
            idx++;
        }
        while (idy < yi.length) {
            yv[idy] = 0;
            idy++;
        }
    }

    private static void hDot(
            SparseVector<Integer, Double, int[], double[]> x,
            DenseVector<Integer, Double, int[], double[]> y) {
        int idx = 0;
        int[] xi = x.getIndices();
        double[] xv = x.getValues();
        double[] yv = y.getValues();
        for (int i = 0; i < y.size(); i++) {
            if (idx < xi.length && xi[idx] == i) {
                yv[i] *= xv[idx];
                idx++;
            } else {
                yv[i] = 0;
            }
        }
    }

    private static void hDot(
            DenseVector<Integer, Double, int[], double[]> x,
            SparseVector<Integer, Double, int[], double[]> y) {
        double[] xv = x.getValues();
        int[] yi = y.getIndices();
        double[] yv = y.getValues();
        for (int i = 0; i < yv.length; i++) {
            yv[i] *= xv[yi[i]];
        }
    }

    private static void hDot(
            DenseVector<Integer, Double, int[], double[]> x,
            DenseVector<Integer, Double, int[], double[]> y) {
        double[] xv = x.getValues();
        double[] yv = y.getValues();
        for (int i = 0; i < xv.length; i++) {
            yv[i] *= xv[i];
        }
    }
}
