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

import static org.apache.flink.ml.common.ps.message.MessageType.PULL_INDEX;

/** The indices one worker needs to pull from servers. */
public class PullIndexM implements Message {
    public final int serverId;
    public final int workerId;
    public final long[] indices;

    public PullIndexM(int serverId, int workerId, long[] indices) {
        this.serverId = serverId;
        this.workerId = workerId;
        this.indices = indices;
    }

    public static PullIndexM fromBytes(byte[] bytes) {
        int offset = 0;
        char type = Bits.getChar(bytes, offset);
        offset += Character.BYTES;
        Preconditions.checkState(type == PULL_INDEX.type);

        int psId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        int workerId = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;
        long[] indices = MessageUtils.getLongArray(bytes, offset);
        return new PullIndexM(psId, workerId, indices);
    }

    @Override
    public byte[] toBytes() {
        int numBytes =
                Character.BYTES + Integer.BYTES * 2 + MessageUtils.getLongArraySizeInBytes(indices);
        byte[] buffer = new byte[numBytes];
        int offset = 0;

        Bits.putChar(buffer, offset, PULL_INDEX.type);
        offset += Character.BYTES;
        Bits.putInt(buffer, offset, this.serverId);
        offset += Integer.BYTES;
        Bits.putInt(buffer, offset, this.workerId);
        offset += Integer.BYTES;
        MessageUtils.putLongArray(this.indices, buffer, offset);
        return buffer;
    }
}
