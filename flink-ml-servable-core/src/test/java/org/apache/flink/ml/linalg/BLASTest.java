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
        // Tests axpy(dense, dense).
        DenseVector anotherDenseVec = Vectors.dense(1, 2, 3, 4, 5);
        BLAS.axpy(1, inputDenseVec, anotherDenseVec);
        double[] expectedResult = new double[] {2, 0, 6, 8, 0};
        assertArrayEquals(expectedResult, anotherDenseVec.values, TOLERANCE);

        // Tests axpy(sparse, dense).
        SparseVector sparseVec = Vectors.sparse(5, new int[] {0, 2, 4}, new double[] {1, 3, 5});
        BLAS.axpy(2, sparseVec, anotherDenseVec);
        expectedResult = new double[] {4, 0, 12, 8, 10};
        assertArrayEquals(expectedResult, anotherDenseVec.values, TOLERANCE);
    }

    @Test
    public void testAxpyK() {
        // Tests axpy(dense, dense, k).
        DenseVector anotherDenseVec = Vectors.dense(1, 2, 3);
        BLAS.axpy(1, inputDenseVec, anotherDenseVec, 3);
        double[] expectedResult = new double[] {2, 0, 6};
        assertArrayEquals(expectedResult, anotherDenseVec.values, TOLERANCE);

        // Tests axpy(sparse, dense, k).
        SparseVector sparseVec = Vectors.sparse(5, new int[] {0, 2, 4}, new double[] {1, 3, 5});
        anotherDenseVec = Vectors.dense(1, 2, 3, 4, 5, 6, 7);
        BLAS.axpy(2, sparseVec, anotherDenseVec, 5);
        expectedResult = new double[] {3, 2, 9, 4, 15, 6, 7};
        assertArrayEquals(expectedResult, anotherDenseVec.values, TOLERANCE);
    }

    @Test
    public void testDot() {
        DenseVector anotherDenseVec = Vectors.dense(1, 2, 3, 4, 5);
        SparseVector sparseVector1 =
                Vectors.sparse(5, new int[] {1, 2, 4}, new double[] {1., 1., 4.});
        SparseVector sparseVector2 =
                Vectors.sparse(5, new int[] {1, 3, 4}, new double[] {1., 2., 1.});
        // Tests dot(dense, dense).
        assertEquals(-3, BLAS.dot(inputDenseVec, anotherDenseVec), TOLERANCE);
        // Tests dot(dense, sparse).
        assertEquals(-19, BLAS.dot(inputDenseVec, sparseVector1), TOLERANCE);
        // Tests dot(sparse, dense).
        assertEquals(1, BLAS.dot(sparseVector2, inputDenseVec), TOLERANCE);
        // Tests dot(sparse, sparse).
        assertEquals(5, BLAS.dot(sparseVector1, sparseVector2), TOLERANCE);
    }

    @Test
    public void testNorm2() {
        assertEquals(Math.sqrt(55), BLAS.norm2(inputDenseVec), TOLERANCE);

        SparseVector sparseVector = Vectors.sparse(5, new int[] {0, 2, 4}, new double[] {1, 3, 5});
        assertEquals(Math.sqrt(35), BLAS.norm2(sparseVector), TOLERANCE);
    }

    @Test
    public void testNorm() {
        assertEquals(Math.sqrt(55), BLAS.norm(inputDenseVec, 2.0), TOLERANCE);

        SparseVector sparseVector = Vectors.sparse(5, new int[] {0, 2, 4}, new double[] {1, 3, 5});
        assertEquals(5.0, BLAS.norm(sparseVector, Double.POSITIVE_INFINITY), TOLERANCE);

        assertEquals(5.348481241239363, BLAS.norm(sparseVector, 3.0), TOLERANCE);
    }

    @Test
    public void testScal() {
        BLAS.scal(2, inputDenseVec);

        double[] expectedDenseResult = new double[] {2, -4, 6, 8, -10};
        assertArrayEquals(expectedDenseResult, inputDenseVec.values, TOLERANCE);

        SparseVector inputSparseVector =
                Vectors.sparse(5, new int[] {0, 2, 4}, new double[] {1, 3, 5});
        BLAS.scal(1.5, inputSparseVector);

        double[] expectedSparseResult = new double[] {1.5, 4.5, 7.5};
        int[] expectedSparseIndices = new int[] {0, 2, 4};

        assertArrayEquals(expectedSparseResult, inputSparseVector.values, TOLERANCE);
        assertArrayEquals(expectedSparseIndices, inputSparseVector.indices);
    }

    @Test
    public void testGemv() {
        DenseVector anotherDenseVec = Vectors.dense(1.0, 2.0);
        BLAS.gemv(-2.0, inputDenseMat, false, inputDenseVec, 0.0, anotherDenseVec);
        double[] expectedResult = new double[] {96.0, -60.0};
        assertArrayEquals(expectedResult, anotherDenseVec.values, TOLERANCE);
    }

    @Test
    public void testHDot() {
        // Tests hDot(sparse, sparse).
        SparseVector sparseVec1 = Vectors.sparse(5, new int[] {0, 2, 3}, new double[] {1, 3, 5});
        SparseVector sparseVec2 = Vectors.sparse(5, new int[] {0, 1, 4}, new double[] {1, 3, 5});
        BLAS.hDot(sparseVec1, sparseVec2);
        assertEquals(5, sparseVec2.size());
        assertArrayEquals(new int[] {0, 1, 4}, sparseVec2.indices);
        assertArrayEquals(new double[] {1, 0, 0}, sparseVec2.values, TOLERANCE);

        // Tests hDot(dense, dense).
        DenseVector denseVec1 = Vectors.dense(1, 2, 3, 4, 5);
        DenseVector denseVec2 = Vectors.dense(1, 2, 3, 4, 5);
        BLAS.hDot(denseVec1, denseVec2);
        double[] expectedResult = new double[] {1, 4, 9, 16, 25};
        assertArrayEquals(expectedResult, denseVec2.values, TOLERANCE);

        // Tests hDot(sparse, dense).
        BLAS.hDot(sparseVec1, denseVec1);
        expectedResult = new double[] {1, 0, 9, 20, 0};
        assertArrayEquals(expectedResult, denseVec1.values, TOLERANCE);

        // Tests hDot(dense, sparse).
        DenseVector denseVec3 = Vectors.dense(1, 2, 3, 4, 5);
        BLAS.hDot(denseVec3, sparseVec1);
        assertEquals(5, sparseVec1.size());
        assertArrayEquals(new int[] {0, 2, 3}, sparseVec1.indices);
        assertArrayEquals(new double[] {1, 9, 20}, sparseVec1.values, TOLERANCE);
    }
}
