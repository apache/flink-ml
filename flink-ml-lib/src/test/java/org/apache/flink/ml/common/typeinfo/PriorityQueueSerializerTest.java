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

package org.apache.flink.ml.common.typeinfo;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.types.Row;
import org.apache.flink.util.TestLogger;

import org.junit.Test;

import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.io.Serializable;
import java.util.Comparator;
import java.util.PriorityQueue;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

/** Tests the serialization and deserialization for the {@link java.util.PriorityQueue}. */
public class PriorityQueueSerializerTest extends TestLogger {

    @Test
    public void testSerializationDeserialization() throws IOException {
        final MockComparator comparator = new MockComparator();
        TypeInformation<PriorityQueue<Row>> typeInfo =
                new PriorityQueueTypeInfo<>(comparator, Types.ROW(Types.DOUBLE, Types.STRING));
        TypeSerializer<PriorityQueue<Row>> serializer =
                typeInfo.createSerializer(new ExecutionConfig());
        assertSame(PriorityQueueSerializer.class, serializer.getClass());

        PipedInputStream pipedInput = new PipedInputStream(1024 * 1024);
        DataInputViewStreamWrapper reader = new DataInputViewStreamWrapper(pipedInput);
        DataOutputViewStreamWrapper writer =
                new DataOutputViewStreamWrapper(new PipedOutputStream(pipedInput));

        PriorityQueue<Row> queue = new PriorityQueue<>(comparator);
        queue.add(Row.of(2.0, "b"));
        queue.add(Row.of(1.0, "a"));
        queue.add(Row.of(3.0, "c"));

        serializer.serialize(queue, writer);
        writer.write(reader, writer.size());

        PriorityQueue<Row> deserialized = serializer.deserialize(reader);

        assertEquals(3, deserialized.size());
        assertEquals("a", deserialized.peek().getFieldAs(1));
    }

    private static class MockComparator implements Comparator<Row>, Serializable {
        @Override
        public int compare(Row o1, Row o2) {
            return Double.compare(o1.getFieldAs(0), o2.getFieldAs(0));
        }
    }
}
