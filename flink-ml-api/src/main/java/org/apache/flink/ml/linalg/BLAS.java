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
    /**
     * For level-2 and level-3 routines, we use the native BLAS.
     *
     * <p>The NATIVE_BLAS instance tries to load BLAS implementations in the order: 1) optimized
     * system libraries such as Intel MKL, 2) self-contained native builds using the reference
     * Fortran from netlib.org, 3) F2J implementation. If to use optimized system libraries, it is
     * important to turn of their multi-thread support. Otherwise, it will conflict with Flink's
     * executor and leads to performance loss.
     */
    private static final dev.ludovic.netlib.BLAS F2J_BLAS =
            dev.ludovic.netlib.JavaBLAS.getInstance();

    /** y += a * x . */
    public static void axpy(double a, double[] x, double[] y) {
        Preconditions.checkArgument(x.length == y.length, "Array dimension mismatched.");
        F2J_BLAS.daxpy(x.length, a, x, 1, y, 1);
    }
}

