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

package org.apache.flink.ml.common.ps.message;

import org.apache.flink.ml.util.Bits;
import org.apache.flink.util.Preconditions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.function.BiFunction;

import static org.apache.flink.ml.common.ps.message.MessageType.ALL_REDUCE_VALUE;

/** The message to apply all-reduce among workers. */
public class AllReduceM implements Message {
    public final int serverId;
    public final int workerId;
    public final double[] values;
    public final BiFunction<double[], double[], double[]> aggregator;

    public AllReduceM(
            int serverId,
            int workerId,
            double[] values,
            BiFunction<double[], double[], double[]> aggregator) {
        this.serverId = serverId;
        this.workerId = workerId;
        this.values = values;
        this.aggregator = aggregator;
    }

    public static AllReduceM fromBytes(byte[] bytes) {
        int offset = 0;
        char type = Bits.getChar(bytes, offset);
        offset += Character.BYTES;
        Preconditions.checkState(type == ALL_REDUCE_VALUE.type);

        int psId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        int workerId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        double[] values = MessageUtils.getDoubleArray(bytes, offset);
        offset += MessageUtils.getDoubleArraySizeInBytes(values);

        BiFunction<double[], double[], double[]> aggregator = deserializeFunction(bytes, offset);
        return new AllReduceM(psId, workerId, values, aggregator);
    }

    @Override
    public byte[] toBytes() {
        byte[] serializedFunctionInBytes = serializeFunction(aggregator);
        int numBytes =
                Character.BYTES
                        + Integer.BYTES
                        + Integer.BYTES
                        + MessageUtils.getDoubleArraySizeInBytes(values)
                        + serializedFunctionInBytes.length;
        byte[] buffer = new byte[numBytes];
        int offset = 0;
        Bits.putChar(buffer, offset, ALL_REDUCE_VALUE.type);
        offset += Character.BYTES;

        Bits.putInt(buffer, offset, this.serverId);
        offset += Integer.BYTES;
        Bits.putInt(buffer, offset, this.workerId);
        offset += Integer.BYTES;
        MessageUtils.putDoubleArray(values, buffer, offset);
        offset += MessageUtils.getDoubleArraySizeInBytes(values);
        System.arraycopy(
                serializedFunctionInBytes, 0, buffer, offset, serializedFunctionInBytes.length);

        return buffer;
    }

    private static byte[] serializeFunction(BiFunction<double[], double[], double[]> aggregator) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try {
            ObjectOutputStream oos = new ObjectOutputStream(baos);
            oos.writeObject(aggregator);
            oos.flush();
        } catch (Throwable e) {
            return null;
        }
        return baos.toByteArray();
    }

    private static BiFunction<double[], double[], double[]> deserializeFunction(
            byte[] bytes, int offset) {
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes, offset, bytes.length - offset);
        try {
            ObjectInputStream ois = new ObjectInputStream(bais);
            return (BiFunction<double[], double[], double[]>) ois.readObject();
        } catch (Exception e) {
            System.out.println("wrong deserialization");
            return null;
        }
    }
}
