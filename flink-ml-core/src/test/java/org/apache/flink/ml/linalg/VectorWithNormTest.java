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

import static org.junit.Assert.assertEquals;

/** Tests the behavior of {@link VectorWithNorm}. */
public class VectorWithNormTest {
    @Test
    public void testL2Norm() {
        DenseVector denseVector = Vectors.dense(1, 2, 3);
        VectorWithNorm denseVectorWithNorm = new VectorWithNorm(denseVector);
        assertEquals(denseVector, denseVectorWithNorm.vector);
        assertEquals(Math.sqrt(14), denseVectorWithNorm.l2Norm, 1e-7);

        SparseVector sparseVector = Vectors.sparse(5, new int[] {0, 2, 4}, new double[] {1, 2, 3});
        VectorWithNorm sparseVectorWithNorm = new VectorWithNorm(sparseVector);
        assertEquals(sparseVector, sparseVectorWithNorm.vector);
        assertEquals(Math.sqrt(14), sparseVectorWithNorm.l2Norm, 1e-7);
    }
}
