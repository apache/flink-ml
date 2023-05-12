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

/** The indices one worker needs to pull from servers. */
public class IndicesToPullM implements Message {
    public final int serverId;
    public final int workerId;
    public final long[] indicesToPull;

    public static final MessageType MESSAGE_TYPE = MessageType.INDICES_TO_PULL;

    public IndicesToPullM(int serverId, int workerId, long[] indicesToPull) {
        this.serverId = serverId;
        this.workerId = workerId;
        this.indicesToPull = indicesToPull;
    }

    public static IndicesToPullM fromBytes(byte[] bytesData) {
        int offset = 0;
        char type = Bits.getChar(bytesData, offset);
        offset += Character.BYTES;
        Preconditions.checkState(type == MESSAGE_TYPE.type);

        int psId = Bits.getInt(bytesData, offset);
        offset += Integer.BYTES;
        int workerId = Bits.getInt(bytesData, offset);
        offset += Integer.BYTES;
        long[] toPullIndices = MessageUtils.readLongArray(bytesData, offset);
        return new IndicesToPullM(psId, workerId, toPullIndices);
    }

    @Override
    public byte[] toBytes() {
        int numBytes =
                Character.BYTES
                        + Integer.BYTES * 2
                        + MessageUtils.getLongArraySizeInBytes(indicesToPull);
        byte[] buffer = new byte[numBytes];
        int offset = 0;

        Bits.putChar(buffer, offset, MESSAGE_TYPE.type);
        offset += Character.BYTES;
        Bits.putInt(buffer, offset, this.serverId);
        offset += Integer.BYTES;
        Bits.putInt(buffer, offset, this.workerId);
        offset += Integer.BYTES;
        MessageUtils.writeLongArray(this.indicesToPull, buffer, offset);
        return buffer;
    }
}
