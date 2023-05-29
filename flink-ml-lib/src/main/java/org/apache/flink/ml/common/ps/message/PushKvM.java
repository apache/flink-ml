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
import org.apache.flink.util.Preconditions;

import static org.apache.flink.ml.common.ps.message.MessageType.PUSH_KV;

/** The sparse key-values to push from workers to servers. */
public class KVsToPushM implements Message {
    public final int serverId;
    public final int workerId;
    public final Tuple2<long[], double[]> kvs;

    public KVsToPushM(int workerId, int serverId, Tuple2<long[], double[]> kvs) {
        this.workerId = workerId;
        this.serverId = serverId;
        this.kvs = kvs;
    }

    public static KVsToPushM fromBytes(byte[] bytes) {
        int offset = 0;
        char type = Bits.getChar(bytes, offset);
        offset += Character.BYTES;
        Preconditions.checkState(type == PUSH_KV.type);

        int workerId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        int psId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        Tuple2<long[], double[]> grad = MessageUtils.readLongDoubleArray(bytes, offset);
        return new KVsToPushM(workerId, psId, grad);
    }

    @Override
    public byte[] toBytes() {
        int numBytes =
                Character.BYTES
                        + Integer.BYTES
                        + Integer.BYTES
                        + MessageUtils.getLongDoubleArraySizeInBytes(kvs);
        byte[] buffer = new byte[numBytes];
        int offset = 0;

        Bits.putChar(buffer, offset, PUSH_KV.type);
        offset += Character.BYTES;

        Bits.putInt(buffer, offset, this.workerId);
        offset += Integer.BYTES;
        Bits.putInt(buffer, offset, this.serverId);
        offset += Integer.BYTES;
        MessageUtils.writeLongDoubleArray(kvs, buffer, offset);

        return buffer;
    }
}
