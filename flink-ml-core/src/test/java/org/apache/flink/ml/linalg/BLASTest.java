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

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests the {@link BLAS}. */
public class BLASTest {

    private static final double TOLERANCE = 1e-7;
    private static final DenseVector inputDenseVec = Vectors.dense(1, -2, 3, 4, -5);
    private static final DenseMatrix inputDenseMat =
            new DenseMatrix(2, 5, new double[] {1, -2, 3, 4, -5, 1, -2, 3, 4, -5});

    @Test
    public void testAsum() {
        assertEquals(15, BLAS.asum(inputDenseVec), TOLERANCE);
    }

    @Test
    public void testAxpy() {
        DenseVector anotherDenseVec = Vectors.dense(1, 2, 3, 4, 5);
        BLAS.axpy(1, inputDenseVec, anotherDenseVec);
        double[] expectedResult = new double[] {2, 0, 6, 8, 0};
        assertArrayEquals(expectedResult, anotherDenseVec.values, TOLERANCE);
    }

    @Test
    public void testDot() {
        DenseVector anotherDenseVec = Vectors.dense(1, 2, 3, 4, 5);
        assertEquals(-3, BLAS.dot(inputDenseVec, anotherDenseVec), TOLERANCE);
    }

    @Test
    public void testNorm2() {
        double expectedResult = Math.sqrt(55);
        assertEquals(expectedResult, BLAS.norm2(inputDenseVec), TOLERANCE);
    }

    @Test
    public void testScal() {
        BLAS.scal(2, inputDenseVec);
        double[] expectedResult = new double[] {2, -4, 6, 8, -10};
        assertArrayEquals(expectedResult, inputDenseVec.values, TOLERANCE);
    }

    @Test
    public void testGemv() {
        DenseVector anotherDenseVec = Vectors.dense(1.0, 2.0);
        BLAS.gemv(-2.0, inputDenseMat, false, inputDenseVec, 0.0, anotherDenseVec);
        double[] expectedResult = new double[] {96.0, -60.0};
        assertArrayEquals(expectedResult, anotherDenseVec.values, TOLERANCE);
    }
}
