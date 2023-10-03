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

import java.util.Arrays;

/**
 * Solver used to solve nonnegative least squares problems using a modified projected gradient
 * method.
 */
public class NNLS {
    private static final dev.ludovic.netlib.BLAS NATIVE_BLAS =
            dev.ludovic.netlib.JavaBLAS.getInstance();

    private double[] scratch;
    private double[] grad;
    private double[] x;
    private double[] dir;
    private double[] lastDir;
    private double[] res;
    private int n;
    private boolean initialized = false;

    public void initialize(int n) {
        if (!initialized) {
            this.n = n;
            scratch = new double[n];
            grad = new double[n];
            x = new double[n];
            dir = new double[n];
            lastDir = new double[n];
            res = new double[n];
            initialized = true;
        }
    }

    public void wipe() {
        Arrays.fill(scratch, 0.0);
        Arrays.fill(grad, 0.0);
        Arrays.fill(x, 0.0);
        Arrays.fill(dir, 0.0);
        Arrays.fill(lastDir, 0.0);
        Arrays.fill(res, 0.0);
    }

    public double[] solve(double[] ata, double[] atb) {
        wipe();

        int n = atb.length;

        int iterMax = Math.max(400, 20 * n);
        double lastNorm = 0.0;
        int iterno = 0;
        int lastWall = 0; // Last iteration when we hit a bound constraint.
        int i;
        while (iterno < iterMax) {
            // find the residual
            NATIVE_BLAS.dgemv("N", n, n, 1.0, ata, n, x, 1, 0.0, res, 1);
            NATIVE_BLAS.daxpy(n, -1.0, atb, 1, res, 1);
            NATIVE_BLAS.dcopy(n, res, 1, grad, 1);

            // project the gradient
            i = 0;
            while (i < n) {
                if (grad[i] > 0.0 && x[i] == 0.0) {
                    grad[i] = 0.0;
                }
                i = i + 1;
            }
            double ngrad = NATIVE_BLAS.ddot(n, grad, 1, grad, 1);

            NATIVE_BLAS.dcopy(n, grad, 1, dir, 1);

            // use a CG direction under certain conditions
            double step = steplen(grad, res, ata);
            double ndir;
            double nx = NATIVE_BLAS.ddot(n, x, 1, x, 1);
            if (iterno > lastWall + 1) {
                double alpha = ngrad / lastNorm;
                NATIVE_BLAS.daxpy(n, alpha, lastDir, 1, dir, 1);
                double dstep = steplen(dir, res, ata);
                ndir = NATIVE_BLAS.ddot(n, dir, 1, dir, 1);
                if (stop(dstep, ndir, nx)) {
                    // reject the CG step if it could lead to premature termination
                    NATIVE_BLAS.dcopy(n, grad, 1, dir, 1);
                    ndir = NATIVE_BLAS.ddot(n, dir, 1, dir, 1);
                } else {
                    step = dstep;
                }
            } else {
                ndir = NATIVE_BLAS.ddot(n, dir, 1, dir, 1);
            }

            // terminate or not.
            if (stop(step, ndir, nx)) {
                return x.clone();
            }

            // don't run through the walls
            i = 0;
            while (i < n) {
                if (step * dir[i] > x[i]) {
                    step = x[i] / dir[i];
                }
                i = i + 1;
            }

            // take the step
            i = 0;
            while (i < n) {
                if (step * dir[i] > x[i] * (1 - 1e-14)) {
                    x[i] = 0;
                    lastWall = iterno;
                } else {
                    x[i] -= step * dir[i];
                }
                i = i + 1;
            }

            iterno = iterno + 1;
            NATIVE_BLAS.dcopy(n, dir, 1, lastDir, 1);
            lastNorm = ngrad;
        }
        return x.clone();
    }

    // find the optimal unconstrained step
    private double steplen(double[] dir, double[] res, double[] ata) {
        double top = NATIVE_BLAS.ddot(n, dir, 1, res, 1);
        NATIVE_BLAS.dgemv("N", n, n, 1.0, ata, n, dir, 1, 0.0, scratch, 1);
        // Push the denominator upward very slightly to avoid infinities and silliness
        return top / (NATIVE_BLAS.ddot(n, scratch, 1, dir, 1) + 1e-20);
    }

    // stopping condition
    boolean stop(Double step, double ndir, double nx) {
        return ((step.isNaN()) // NaN
                || (step < 1e-7) // too small or negative
                || (step > 1e40) // too small; almost certainly numerical problems
                || (ndir < 1e-12 * nx) // gradient relatively too small
                || (ndir < 1e-32) // gradient absolutely too small; numerical issues may lurk
        );
    }
}
