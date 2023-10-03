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

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link org.apache.flink.ml.common.ps.Message}. */
public class MessageTest {
    private Message messageFromBytes;
    private Message messageFromArray;
    private Message messageFromPojo;

    private TypeSerializer<MockPojo> mockPojoTypeSerializer;

    @Before
    public void before() throws IOException {
        messageFromArray = new Message(1, 0, 1, new long[] {1, 2}, new double[] {1, 2, 3, 4});
        messageFromBytes = new Message(messageFromArray.bytes.clone());
        mockPojoTypeSerializer = Types.POJO(MockPojo.class).createSerializer(new ExecutionConfig());
        messageFromPojo =
                new Message(
                        1,
                        0,
                        1,
                        new long[] {1, 2},
                        new MockPojo[] {new MockPojo(1, 1), new MockPojo(2, 2)},
                        mockPojoTypeSerializer);
    }

    @Test
    public void getKeys() {
        long[] expectedKeys = new long[] {1, 2};
        assertArrayEquals(expectedKeys, messageFromArray.getKeys());
        assertArrayEquals(expectedKeys, messageFromBytes.getKeys());
        assertArrayEquals(expectedKeys, messageFromPojo.getKeys());
    }

    @Test
    public void getValuesInDoubleArray() {
        double[] expectedDoubleArray = new double[] {1, 2, 3, 4};
        assertArrayEquals(expectedDoubleArray, messageFromArray.getValuesInDoubleArray(), 1e-7);
        assertArrayEquals(expectedDoubleArray, messageFromBytes.getValuesInDoubleArray(), 1e-7);
    }

    @Test
    public void getValues() throws IOException {
        MockPojo[] expectedPojos = new MockPojo[] {new MockPojo(1, 1), new MockPojo(2, 2)};
        assertArrayEquals(expectedPojos, messageFromPojo.getValues(mockPojoTypeSerializer));
    }

    @Test
    public void getWorkerId() {
        int expectedWorkerId = 1;
        assertEquals(expectedWorkerId, messageFromArray.getWorkerId());
        assertEquals(expectedWorkerId, messageFromBytes.getWorkerId());
        assertEquals(expectedWorkerId, messageFromPojo.getWorkerId());
    }

    @Test
    public void setWorkerId() {
        messageFromArray.setWorkerId(2);
        messageFromBytes.setWorkerId(2);
        messageFromPojo.setWorkerId(2);
        int expectedWorkerId = 2;
        assertEquals(expectedWorkerId, messageFromArray.getWorkerId());
        assertEquals(expectedWorkerId, messageFromBytes.getWorkerId());
        assertEquals(expectedWorkerId, messageFromPojo.getWorkerId());
    }

    @Test
    public void getServerId() {
        int expectedServerId = 0;
        assertEquals(expectedServerId, messageFromArray.getServerId());
        assertEquals(expectedServerId, messageFromBytes.getServerId());
        assertEquals(expectedServerId, messageFromPojo.getServerId());
    }

    @Test
    public void setServerId() {
        messageFromArray.setServerId(2);
        messageFromBytes.setServerId(2);
        messageFromPojo.setServerId(2);
        int expectedServerId = 2;
        assertEquals(expectedServerId, messageFromArray.getServerId());
        assertEquals(expectedServerId, messageFromBytes.getServerId());
        assertEquals(expectedServerId, messageFromPojo.getServerId());
    }

    @Test
    public void getStagedId() {
        int expectedStageId = 1;
        assertEquals(expectedStageId, messageFromArray.getStageId());
        assertEquals(expectedStageId, messageFromBytes.getStageId());
        assertEquals(expectedStageId, messageFromPojo.getStageId());
    }

    @Test
    public void assembleMessages() {
        int numServers = 4;
        Message[] messages = new Message[numServers];
        for (int i = 0; i < numServers; i++) {
            messages[i] =
                    new Message(
                            1,
                            i,
                            0,
                            new long[] {i * 2, i * 2 + 1},
                            new double[] {i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3});
        }

        Iterator<byte[]> bytes = Arrays.stream(messages).map(x -> x.bytes).iterator();
        Message assembledMessage = Message.assembleMessages(bytes);

        assertEquals(-1, assembledMessage.getServerId());
        assertEquals(1, assembledMessage.getWorkerId());
        assertEquals(0, assembledMessage.getStageId());

        long[] expectedKeys = new long[numServers * 2];
        for (int i = 0; i < expectedKeys.length; i++) {
            expectedKeys[i] = i;
        }
        assertArrayEquals(expectedKeys, assembledMessage.getKeys());

        double[] expectedValues = new double[numServers * 4];
        for (int i = 0; i < expectedValues.length; i++) {
            expectedValues[i] = i;
        }
        assertArrayEquals(expectedValues, assembledMessage.getValuesInDoubleArray(), 1e-7);
    }
}
