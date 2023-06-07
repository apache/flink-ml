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

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.util.Bits;
import org.apache.flink.util.Preconditions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * {@link Message} is responsible for encoding all messages exchanged between a worker and a server.
 * The message format follows this structure:
 *
 * <p>`workerId serverId messageType keyLength keys valuesLength values`
 *
 * <p>where the message fields include the worker ID, server ID, message type, length of the keys,
 * keys themselves, length of the values, and the values.
 */
public class Message {
    private static final int WORKER_ID_OFFSET = 0;
    private static final int SERVER_ID_OFFSET = Integer.BYTES;
    private static final int MESSAGE_TYPE_OFFSET = Integer.BYTES + SERVER_ID_OFFSET;
    private static final int KVS_OFFSET = Integer.BYTES + MESSAGE_TYPE_OFFSET;

    public final byte[] bytes;

    public Message(byte[] bytes) {
        this.bytes = bytes;
    }

    /** Constructs a message instance from long keys and double values. */
    public Message(
            int serverId, int workerId, MessageType messageType, long[] keys, double[] values) {
        int sizeInBytes = KVS_OFFSET + Bits.getLongDoubleArraySizeInBytes(Tuple2.of(keys, values));
        bytes = new byte[sizeInBytes];
        Bits.putInt(bytes, WORKER_ID_OFFSET, workerId);
        Bits.putInt(bytes, SERVER_ID_OFFSET, serverId);
        Bits.putInt(bytes, MESSAGE_TYPE_OFFSET, messageType.type);
        Bits.putLongDoubleArray(Tuple2.of(keys, values), bytes, KVS_OFFSET);
    }

    /** Constructs a message instance from long keys and generics values. */
    public <V> Message(
            int serverId,
            int workerId,
            MessageType messageType,
            long[] keys,
            V[] values,
            TypeSerializer<V> serializer)
            throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        DataOutputViewStreamWrapper dataOutputViewStreamWrapper =
                new DataOutputViewStreamWrapper(byteArrayOutputStream);
        dataOutputViewStreamWrapper.writeInt(workerId);
        dataOutputViewStreamWrapper.writeInt(serverId);
        dataOutputViewStreamWrapper.writeInt(messageType.type);

        dataOutputViewStreamWrapper.writeInt(keys.length);
        for (long key : keys) {
            dataOutputViewStreamWrapper.writeLong(key);
        }
        dataOutputViewStreamWrapper.writeInt(values.length);
        for (V value : values) {
            serializer.serialize(value, dataOutputViewStreamWrapper);
        }
        bytes = byteArrayOutputStream.toByteArray();
    }

    /** Retrieves the keys. */
    public long[] getKeys() {
        return Bits.getLongArray(bytes, KVS_OFFSET);
    }

    /** Retrieves the values using the given serializer. */
    public <V> V[] getValues(TypeSerializer<V> serializer) throws IOException {
        int numIndices = Bits.getInt(bytes, KVS_OFFSET);
        int offset = KVS_OFFSET + Integer.BYTES + numIndices * Long.BYTES;
        int numValues = Bits.getInt(bytes, offset);
        offset += Integer.BYTES;

        // Since the generics got erased, we use reflections to create the array.
        V[] result = (V[]) Array.newInstance(serializer.createInstance().getClass(), numValues);
        ByteArrayInputStream byteArrayInputStream =
                new ByteArrayInputStream(bytes, offset, bytes.length - offset);
        DataInputViewStreamWrapper dataInputViewStreamWrapper =
                new DataInputViewStreamWrapper(byteArrayInputStream);
        for (int i = 0; i < numValues; i++) {
            result[i] = serializer.deserialize(dataInputViewStreamWrapper);
        }
        return result;
    }

    /** Retrieves the values in double array format. */
    public double[] getValuesInDoubleArray() {
        int offset = KVS_OFFSET + Bits.getInt(bytes, KVS_OFFSET) * Long.BYTES + Integer.BYTES;
        return Bits.getDoubleArray(bytes, offset);
    }

    /** Retrieves the worker id. */
    public int getWorkerId() {
        return Bits.getInt(bytes, WORKER_ID_OFFSET);
    }

    /** Retrieves the server id. */
    public int getServerId() {
        return Bits.getInt(bytes, SERVER_ID_OFFSET);
    }

    /** Sets the worker id. */
    public void setWorkerId(int workerId) {
        Bits.putInt(bytes, WORKER_ID_OFFSET, workerId);
    }

    /** Sets the server id. */
    public void setServerId(int serverId) {
        Bits.putInt(bytes, SERVER_ID_OFFSET, serverId);
    }

    /** Retrieves the message type. */
    public MessageType getMessageType() {
        return MessageType.valueOf(Bits.getInt(bytes, MESSAGE_TYPE_OFFSET));
    }

    /**
     * Assembles the received messages from servers according to the server id. Note that these
     * messages should come from the same request.
     */
    public static Message assembleMessages(Iterator<byte[]> messageIterator) {
        List<Message> messages = new ArrayList<>();
        while (messageIterator.hasNext()) {
            messages.add(new Message(messageIterator.next()));
        }
        messages.sort(Comparator.comparingInt(Message::getServerId));

        int numMessages = messages.size();
        int numKeys = 0, numValues = 0;
        int numAssembledBytes = 0;
        int workerId = -1;
        for (Message message : messages) {
            Preconditions.checkState(workerId == -1 || workerId == message.getWorkerId());
            workerId = message.getWorkerId();
            numKeys += message.getNumKeys();
            numValues += message.getNumValues();
            numAssembledBytes += message.bytes.length;
        }
        numAssembledBytes -= (numMessages - 1) * (KVS_OFFSET + Integer.BYTES * 2);
        byte[] assembledBytes = new byte[numAssembledBytes];
        int keysOffset = KVS_OFFSET;
        Bits.putInt(assembledBytes, keysOffset, numKeys);
        keysOffset += Integer.BYTES;
        int valuesOffset = keysOffset + numKeys * Long.BYTES;
        Bits.putInt(assembledBytes, valuesOffset, numValues);
        valuesOffset += Integer.BYTES;

        for (Message message : messages) {
            Tuple2<Integer, Integer> keyOoffsetAndLength = message.getKeysOffsetAndLength();
            System.arraycopy(
                    message.bytes,
                    keyOoffsetAndLength.f0,
                    assembledBytes,
                    keysOffset,
                    keyOoffsetAndLength.f1);
            keysOffset += keyOoffsetAndLength.f1;
            Tuple2<Integer, Integer> valuesOffsetAndLength = message.getValuesOffSetAndLength();
            System.arraycopy(
                    message.bytes,
                    valuesOffsetAndLength.f0,
                    assembledBytes,
                    valuesOffset,
                    valuesOffsetAndLength.f1);
            valuesOffset += valuesOffsetAndLength.f1;
        }

        Message message = new Message(assembledBytes);
        message.setServerId(-1);
        message.setWorkerId(workerId);
        return message;
    }

    private Tuple2<Integer, Integer> getKeysOffsetAndLength() {
        int start = KVS_OFFSET + Integer.BYTES;
        int numBytes = Bits.getInt(bytes, KVS_OFFSET) * Long.BYTES;
        return Tuple2.of(start, numBytes);
    }

    private Tuple2<Integer, Integer> getValuesOffSetAndLength() {
        int start =
                Bits.getInt(bytes, KVS_OFFSET) * Long.BYTES
                        + KVS_OFFSET
                        + Integer.BYTES
                        + Integer.BYTES;
        return Tuple2.of(start, bytes.length - start);
    }

    private int getNumKeys() {
        return Bits.getInt(bytes, KVS_OFFSET);
    }

    private int getNumValues() {
        return Bits.getInt(bytes, KVS_OFFSET + Integer.BYTES + Long.BYTES * getNumKeys());
    }
}
