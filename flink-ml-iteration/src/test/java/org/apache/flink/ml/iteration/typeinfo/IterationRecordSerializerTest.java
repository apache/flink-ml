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

package org.apache.flink.ml.iteration.typeinfo;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.api.common.typeutils.base.VoidSerializer;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.iteration.IterationRecord;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 * Tests the serialization and deserialization for the {@link
 * org.apache.flink.ml.iteration.IterationRecord}
 */
public class IterationRecordSerializerTest {

    @Test
    public void testRecordType() throws IOException {
        testSerializeAndDeserialize(IterationRecord.newRecord(null, 3), StringSerializer.INSTANCE);
        testSerializeAndDeserialize(IterationRecord.newRecord(5, 3), IntSerializer.INSTANCE);
        testSerializeAndDeserialize(
                IterationRecord.newRecord("Best", 3), StringSerializer.INSTANCE);
    }

    @Test
    public void testEpochWatermarkType() throws IOException {
        testSerializeAndDeserialize(
                IterationRecord.newEpochWatermark(10, "sender1"), VoidSerializer.INSTANCE);
        testSerializeAndDeserialize(
                IterationRecord.newEpochWatermark(Integer.MAX_VALUE, "sender1"),
                VoidSerializer.INSTANCE);
    }

    @Test
    public void testBarrierType() throws IOException {
        testSerializeAndDeserialize(IterationRecord.newBarrier(15), VoidSerializer.INSTANCE);
    }

    private static <T> void testSerializeAndDeserialize(
            IterationRecord<T> iterationRecord, TypeSerializer<T> internalSerializer)
            throws IOException {
        IterationRecordSerializer<T> iterationRecordSerializer =
                new IterationRecordSerializer<T>(internalSerializer);

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputView dataOutputView = new DataOutputViewStreamWrapper(bos);
        iterationRecordSerializer.serialize(iterationRecord, dataOutputView);

        byte[] serializedData = bos.toByteArray();
        bos.close();

        ByteArrayInputStream bis = new ByteArrayInputStream(serializedData);
        DataInputView dataInputView = new DataInputViewStreamWrapper(bis);
        IterationRecord<T> deserialized = iterationRecordSerializer.deserialize(dataInputView);

        assertEquals(iterationRecord, deserialized);
    }
}
