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

package org.apache.flink.ml.common.ps.training;

import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.util.Bits;

import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Benchmark for PS-related stuff. */
public class PSBench {
    @Test
    public void benchBits() {
        double[] result = new double[100000];
        int warmUp = 500;
        int numTries = 1000;
        for (int i = 0; i < result.length; i++) {
            result[i] = i;
        }
        byte[] bytes = new byte[Bits.getDoubleArraySizeInBytes(result)];

        for (int i = 0; i < warmUp; i++) {
            Bits.putDoubleArray(result, bytes, 0);
        }

        long start = System.currentTimeMillis();
        for (int i = 0; i < numTries; i++) {
            Bits.putDoubleArray(result, bytes, 0);
        }
        long end = System.currentTimeMillis();
        System.out.println(end - start); // ~600ms
    }

    @Test
    public void benchTypeSerializer() throws IOException {
        double[] result = new double[100000];
        int warmUp = 500;
        int numTries = 1000;
        for (int i = 0; i < result.length; i++) {
            result[i] = i;
        }
        byte[] bytes = new byte[Bits.getDoubleArraySizeInBytes(result)];

        for (int i = 0; i < warmUp; i++) {
            bytes = serializeDoubleArray(result);
        }

        long start = System.currentTimeMillis();
        for (int i = 0; i < numTries; i++) {
            bytes = serializeDoubleArray(result);
        }
        long end = System.currentTimeMillis();
        System.out.println(end - start); // 2000ms
        Assert.assertEquals(Bits.getDoubleArraySizeInBytes(result), bytes.length);
    }

    private byte[] serializeDoubleArray(double[] result) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        DataOutputViewStreamWrapper dataOutputViewStreamWrapper =
                new DataOutputViewStreamWrapper(byteArrayOutputStream);
        dataOutputViewStreamWrapper.writeInt(result.length);

        for (double value : result) {
            dataOutputViewStreamWrapper.writeDouble(value);
        }
        return byteArrayOutputStream.toByteArray();
    }
}
