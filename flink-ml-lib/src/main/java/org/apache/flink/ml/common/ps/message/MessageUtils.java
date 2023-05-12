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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.util.Bits;

/** Utility functions for processing messages. */
public class MessageUtils {

    /** Retrieves the message type from the byte array. */
    public static MessageType getMessageType(byte[] bytesData) {
        char type = Bits.getChar(bytesData, 0);
        return MessageType.valueOf(type);
    }

    /** Reads a long array from the byte array starting from the given offset. */
    public static long[] readLongArray(byte[] bytesData, int offset) {
        int size = Bits.getInt(bytesData, offset);
        offset += Integer.BYTES;
        long[] result = new long[size];
        for (int i = 0; i < size; i++) {
            result[i] = Bits.getLong(bytesData, offset);
            offset += Long.BYTES;
        }
        return result;
    }

    /**
     * Writes a long array to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int writeLongArray(long[] array, byte[] bytesData, int offset) {
        Bits.putInt(bytesData, offset, array.length);
        offset += Integer.BYTES;
        for (int i = 0; i < array.length; i++) {
            Bits.putLong(bytesData, offset, array[i]);
            offset += Long.BYTES;
        }
        return offset;
    }

    /** Returns the size of a long array in bytes. */
    public static int getLongArraySizeInBytes(long[] array) {
        return Integer.BYTES + array.length * Long.BYTES;
    }

    /** Reads a double array from the byte array starting from the given offset. */
    public static double[] readDoubleArray(byte[] bytesData, int offset) {
        int size = Bits.getInt(bytesData, offset);
        offset += Integer.BYTES;
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Bits.getDouble(bytesData, offset);
            offset += Long.BYTES;
        }
        return result;
    }

    /**
     * Writes a double array to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int writeDoubleArray(double[] array, byte[] bytesData, int offset) {
        Bits.putInt(bytesData, offset, array.length);
        offset += Integer.BYTES;
        for (int i = 0; i < array.length; i++) {
            Bits.putDouble(bytesData, offset, array[i]);
            offset += Double.BYTES;
        }
        return offset;
    }

    /** Returns the size of a double array in bytes. */
    public static int getDoubleArraySizeInBytes(double[] array) {
        return Integer.BYTES + array.length * Long.BYTES;
    }

    /** Reads a long-double array from the byte array starting from the given offset. */
    public static Tuple2<long[], double[]> readLongDoubleArray(byte[] bytesData, int offset) {
        long[] indices = readLongArray(bytesData, offset);
        offset += getLongArraySizeInBytes(indices);
        double[] values = readDoubleArray(bytesData, offset);
        return Tuple2.of(indices, values);
    }

    /**
     * Writes a long-double to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int writeLongDoubleArray(
            Tuple2<long[], double[]> longDoubleArray, byte[] bytesData, int offset) {
        offset = writeLongArray(longDoubleArray.f0, bytesData, offset);
        offset = writeDoubleArray(longDoubleArray.f1, bytesData, offset);

        return offset;
    }

    /** Returns the size of a long-double array in bytes. */
    public static int getLongDoubleArraySizeInBytes(Tuple2<long[], double[]> longDoubleArray) {
        return getLongArraySizeInBytes(longDoubleArray.f0)
                + getDoubleArraySizeInBytes(longDoubleArray.f1);
    }
}
