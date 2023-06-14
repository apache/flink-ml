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

import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.typeinfo.SparseLongDoubleVectorSerializer;

import org.apache.commons.io.output.ByteArrayOutputStream;
import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests the behavior of {@link SparseLongDoubleVector}. */
public class SparseLongDoubleVectorTest {
    private static final double TOLERANCE = 1e-7;

    @Test
    public void testConstructor() {
        long n = 4;
        long[] indices = new long[] {0, 2, 3};
        double[] values = new double[] {0.1, 0.3, 0.4};

        SparseVector<Long, Double, long[], double[]> vector = Vectors.sparse(n, indices, values);
        assertEquals(n, vector.size());
        assertArrayEquals(indices, vector.getIndices());
        assertArrayEquals(values, vector.getValues(), 1e-5);
        assertEquals("(4, [0, 2, 3], [0.1, 0.3, 0.4])", vector.toString());
    }

    @Test
    public void testDuplicateIndex() {
        long n = 4;
        long[] indices = new long[] {0, 2, 2};
        double[] values = new double[] {0.1, 0.3, 0.4};

        try {
            Vectors.sparse(n, indices, values);
            Assert.fail("Expected IllegalArgumentException.");
        } catch (Exception e) {
            assertEquals(IllegalArgumentException.class, e.getClass());
            assertEquals("Indices duplicated.", e.getMessage());
        }
    }

    @Test
    public void testAllZeroVector() {
        long n = 4;
        SparseVector<Long, Double, long[], double[]> vector =
                Vectors.sparse(n, new long[0], new double[0]);
        assertArrayEquals(vector.toArray(), new double[(int) n], 1e-5);
    }

    @Test
    public void testUnsortedIndex() {
        SparseVector<Long, Double, long[], double[]> vector;

        vector = Vectors.sparse(4, new long[] {2}, new double[] {0.3});
        assertEquals(4, vector.size());
        assertArrayEquals(new long[] {2}, vector.getIndices());
        assertArrayEquals(new double[] {0.3}, vector.getValues(), 1e-5);

        vector = Vectors.sparse(4, new long[] {1, 2}, new double[] {0.2, 0.3});
        assertEquals(4, vector.size());
        assertArrayEquals(new long[] {1, 2}, vector.getIndices());
        assertArrayEquals(new double[] {0.2, 0.3}, vector.getValues(), 1e-5);

        vector = Vectors.sparse(4, new long[] {2, 1}, new double[] {0.3, 0.2});
        assertEquals(4, vector.size());
        assertArrayEquals(new long[] {1, 2}, vector.getIndices());
        assertArrayEquals(new double[] {0.2, 0.3}, vector.getValues(), 1e-5);

        vector = Vectors.sparse(4, new long[] {3, 2, 0}, new double[] {0.4, 0.3, 0.1});
        assertEquals(4, vector.size());
        assertArrayEquals(new long[] {0, 2, 3}, vector.getIndices());
        assertArrayEquals(new double[] {0.1, 0.3, 0.4}, vector.getValues(), 1e-5);

        vector = Vectors.sparse(4, new long[] {2, 0, 3}, new double[] {0.3, 0.1, 0.4});
        assertEquals(4, vector.size());
        assertArrayEquals(new long[] {0, 2, 3}, vector.getIndices());
        assertArrayEquals(new double[] {0.1, 0.3, 0.4}, vector.getValues(), 1e-5);

        vector =
                Vectors.sparse(
                        7,
                        new long[] {6, 5, 4, 3, 2, 1, 0},
                        new double[] {0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1});
        assertEquals(7, vector.size());
        assertArrayEquals(new long[] {0, 1, 2, 3, 4, 5, 6}, vector.getIndices());
        assertArrayEquals(
                new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}, vector.getValues(), 1e-5);
    }

    @Test
    public void testSerializer() throws IOException {
        long n = 4;
        long[] indices = new long[] {0, 2, 3};
        double[] values = new double[] {0.1, 0.3, 0.4};
        SparseLongDoubleVector vector = Vectors.sparse(n, indices, values);
        SparseLongDoubleVectorSerializer serializer = SparseLongDoubleVectorSerializer.INSTANCE;

        ByteArrayOutputStream bOutput = new ByteArrayOutputStream(1024);
        DataOutputViewStreamWrapper output = new DataOutputViewStreamWrapper(bOutput);
        serializer.serialize(vector, output);

        byte[] b = bOutput.toByteArray();
        ByteArrayInputStream bInput = new ByteArrayInputStream(b);
        DataInputViewStreamWrapper input = new DataInputViewStreamWrapper(bInput);
        SparseLongDoubleVector vector2 = serializer.deserialize(input);

        assertEquals(vector.size(), vector2.size());
        assertArrayEquals(vector.getIndices(), vector2.getIndices());
        assertArrayEquals(vector.getValues(), vector2.getValues(), 1e-5);
    }

    @Test
    public void testClone() {
        SparseVector<Long, Double, long[], double[]> sparseVec =
                Vectors.sparse(3, new long[] {0, 2}, new double[] {1, 3});
        SparseVector<Long, Double, long[], double[]> clonedSparseVec = sparseVec.clone();
        assertEquals(3, clonedSparseVec.size());
        assertArrayEquals(clonedSparseVec.getIndices(), new long[] {0, 2});
        assertArrayEquals(clonedSparseVec.getValues(), new double[] {1, 3}, TOLERANCE);

        clonedSparseVec.set(0L, -1.0);
        assertEquals(sparseVec.size(), clonedSparseVec.size());
        assertArrayEquals(sparseVec.getIndices(), new long[] {0, 2});
        assertArrayEquals(sparseVec.getValues(), new double[] {1, 3}, TOLERANCE);
        assertArrayEquals(clonedSparseVec.getIndices(), new long[] {0, 2});
        assertArrayEquals(clonedSparseVec.getValues(), new double[] {-1, 3}, TOLERANCE);
    }

    @Test
    public void testGetAndSet() {
        SparseVector<Long, Double, long[], double[]> sparseVec =
                Vectors.sparse(4, new long[] {2}, new double[] {0.3});
        assertEquals(0, sparseVec.get(0L), TOLERANCE);
        assertEquals(0.3, sparseVec.get(2L), TOLERANCE);

        sparseVec.set(2L, 0.5);
        assertEquals(0.5, sparseVec.get(2L), TOLERANCE);

        sparseVec.set(0L, 0.1);
        assertEquals(0.1, sparseVec.get(0L), TOLERANCE);
    }
}
