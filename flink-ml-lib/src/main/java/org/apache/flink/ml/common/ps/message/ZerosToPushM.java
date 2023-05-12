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

/**
 * Message sent by worker to server that initializes the model as a dense array with defined range.
 */
public class ZerosToPushM implements Message {
    public final int workerId;
    public final int serverId;
    public final long startIndex;
    public final long endIndex;

    public static final MessageType MESSAGE_TYPE = MessageType.ZEROS_TO_PUSH;

    public ZerosToPushM(int workerId, int serverId, long startIndex, long endIndex) {
        this.workerId = workerId;
        this.serverId = serverId;
        this.startIndex = startIndex;
        this.endIndex = endIndex;
    }

    public static ZerosToPushM fromBytes(byte[] bytesData) {
        int offset = 0;
        char type = Bits.getChar(bytesData, offset);
        offset += Character.BYTES;
        Preconditions.checkState(type == MESSAGE_TYPE.type);

        int workerId = Bits.getInt(bytesData, offset);
        offset += Integer.BYTES;
        int serverId = Bits.getInt(bytesData, offset);
        offset += Integer.BYTES;
        long startIndex = Bits.getLong(bytesData, offset);
        offset += Long.BYTES;
        long endIndex = Bits.getLong(bytesData, offset);
        return new ZerosToPushM(workerId, serverId, startIndex, endIndex);
    }

    @Override
    public byte[] toBytes() {
        int numBytes = Character.BYTES + Integer.BYTES + Integer.BYTES + Long.BYTES + Long.BYTES;
        byte[] buffer = new byte[numBytes];
        int offset = 0;
        Bits.putChar(buffer, offset, MESSAGE_TYPE.type);
        offset += Character.BYTES;

        Bits.putInt(buffer, offset, this.workerId);
        offset += Integer.BYTES;
        Bits.putInt(buffer, offset, this.serverId);
        offset += Integer.BYTES;
        Bits.putLong(buffer, offset, this.startIndex);
        offset += Long.BYTES;
        Bits.putLong(buffer, offset, this.endIndex);

        return buffer;
    }
}
