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

package org.apache.flink.ml.common.util;

import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

/** Tests {@link VectorUtils}. */
public class VectorUtilsTest {

    private static final double EPS = 1.0e-5;

    @Test
    public void testSelectByIndices() {
        DenseIntDoubleVector denseVector = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0);
        assertArrayEquals(
                Vectors.dense(2.0, 4.0).toArray(),
                VectorUtils.selectByIndices(denseVector, new int[] {1, 3}).toArray(),
                EPS);

        SparseIntDoubleVector sparseVector =
                Vectors.sparse(5, new int[] {1, 2, 3}, new double[] {2.0, 3.0, 4.0});
        assertArrayEquals(
                Vectors.sparse(3, new int[] {1, 2}, new double[] {2.0, 4.0}).toArray(),
                VectorUtils.selectByIndices(sparseVector, new int[] {0, 1, 3}).toArray(),
                EPS);
    }
}
