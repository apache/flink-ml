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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.util.Bits;

/** Utility functions for processing messages. */
public class MessageUtils {

    public static <V> TypeInformation getKeyType(V key) {
        if (key instanceof Integer) {
            return Types.INT;
        } else if (key instanceof Long) {
            return Types.LONG;
        } else {
            throw new UnsupportedOperationException(
                    String.format("Unsupported key type: %s.", key.getClass().getSimpleName()));
        }
    }

    /** Retrieves the message type from the byte array. */
    public static MessageType getMessageType(byte[] bytes) {
        char type = Bits.getChar(bytes, 0);
        return MessageType.valueOf(type);
    }

    /** Gets a long array from the byte array starting from the given offset. */
    public static long[] getLongArray(byte[] bytes, int offset) {
        int size = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        long[] result = new long[size];
        for (int i = 0; i < size; i++) {
            result[i] = Bits.getLong(bytes, offset);
            offset += Long.BYTES;
        }
        return result;
    }

    /**
     * Puts a long array to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int putLongArray(long[] array, byte[] bytes, int offset) {
        Bits.putInt(bytes, offset, array.length);
        offset += Integer.BYTES;
        for (int i = 0; i < array.length; i++) {
            Bits.putLong(bytes, offset, array[i]);
            offset += Long.BYTES;
        }
        return offset;
    }

    /** Returns the size of a long array in bytes. */
    public static int getLongArraySizeInBytes(long[] array) {
        return Integer.BYTES + array.length * Long.BYTES;
    }

    /** Gets a double array from the byte array starting from the given offset. */
    public static double[] getDoubleArray(byte[] bytes, int offset) {
        int size = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Bits.getDouble(bytes, offset);
            offset += Long.BYTES;
        }
        return result;
    }

    /**
     * Puts a double array to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int putDoubleArray(double[] array, byte[] bytes, int offset) {
        Bits.putInt(bytes, offset, array.length);
        offset += Integer.BYTES;
        for (int i = 0; i < array.length; i++) {
            Bits.putDouble(bytes, offset, array[i]);
            offset += Double.BYTES;
        }
        return offset;
    }

    /** Returns the size of a double array in bytes. */
    public static int getDoubleArraySizeInBytes(double[] array) {
        return Integer.BYTES + array.length * Long.BYTES;
    }

    /** Gets a long-double array from the byte array starting from the given offset. */
    public static Tuple2<long[], double[]> getLongDoubleArray(byte[] bytes, int offset) {
        long[] indices = getLongArray(bytes, offset);
        offset += getLongArraySizeInBytes(indices);
        double[] values = getDoubleArray(bytes, offset);
        return Tuple2.of(indices, values);
    }

    /**
     * Puts a long-double array to the byte array starting from the given offset.
     *
     * @return the next position to write on.
     */
    public static int putLongDoubleArray(
            Tuple2<long[], double[]> longDoubleArray, byte[] bytes, int offset) {
        offset = putLongArray(longDoubleArray.f0, bytes, offset);
        offset = putDoubleArray(longDoubleArray.f1, bytes, offset);

        return offset;
    }

    /** Returns the size of a long-double array in bytes. */
    public static int getLongDoubleArraySizeInBytes(Tuple2<long[], double[]> longDoubleArray) {
        return getLongArraySizeInBytes(longDoubleArray.f0)
                + getDoubleArraySizeInBytes(longDoubleArray.f1);
    }
}
