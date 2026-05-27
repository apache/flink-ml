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

package org.apache.flink.ml.linalg.typeinfo;

import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;

import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Random;

/**
 * Tests the serialization and deserialization for the double array from {@link
 * OptimizedDoublePrimitiveArraySerializer}.
 */
public class OptimizedDoublePrimitiveArraySerializerTest {
    @Test
    public void testSerializationDeserialization() throws IOException {
        Random random = new Random(0);
        int[] lens = new int[] {0, 100, 128, 500, 1024, 4096};

        OptimizedDoublePrimitiveArraySerializer serializer =
                new OptimizedDoublePrimitiveArraySerializer();
        for (int len : lens) {
            double[] arr = new double[len];
            for (int i = 0; i < len; i += 1) {
                arr[i] = random.nextDouble();
            }

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            serializer.serialize(arr, new DataOutputViewStreamWrapper(baos));
            double[] actual =
                    serializer.deserialize(
                            new DataInputViewStreamWrapper(
                                    new ByteArrayInputStream(baos.toByteArray())));
            Assert.assertArrayEquals(arr, actual, 0.);
        }
    }
}
