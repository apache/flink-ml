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

import static org.apache.flink.ml.common.ps.message.MessageType.INITIALIZE_MODEL_AS_ZERO;

/** Message sent by worker to server that initializes the model as zeros with defined range. */
public class InitializeModelAsZeroM implements Message {
    public final int workerId;
    public final int serverId;
    public final long startIndex;
    public final long endIndex;

    public InitializeModelAsZeroM(int workerId, int serverId, long startIndex, long endIndex) {
        this.workerId = workerId;
        this.serverId = serverId;
        this.startIndex = startIndex;
        this.endIndex = endIndex;
    }

    public static InitializeModelAsZeroM fromBytes(byte[] bytes) {
        int offset = 0;
        char type = Bits.getChar(bytes, offset);
        offset += Character.BYTES;
        Preconditions.checkState(type == INITIALIZE_MODEL_AS_ZERO.type);

        int workerId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        int serverId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        long startIndex = Bits.getLong(bytes, offset);
        offset += Long.BYTES;
        long endIndex = Bits.getLong(bytes, offset);
        return new InitializeModelAsZeroM(workerId, serverId, startIndex, endIndex);
    }

    @Override
    public byte[] toBytes() {
        int numBytes = Character.BYTES + Integer.BYTES + Integer.BYTES + Long.BYTES + Long.BYTES;
        byte[] buffer = new byte[numBytes];
        int offset = 0;
        Bits.putChar(buffer, offset, INITIALIZE_MODEL_AS_ZERO.type);
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
