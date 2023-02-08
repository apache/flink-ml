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
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.typeutils.TypeExtractor;
import org.apache.flink.api.java.typeutils.runtime.PojoSerializer;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.common.util.QuantileSummary;

import org.junit.Test;

import java.io.IOException;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;

/** Tests {@link QuantileSummaryTypeInfoFactory}. */
public class QuantileSummaryTypeInfoFactoryTest {

    private static final double EPS = 1.0e-5;

    @Test
    public void testSerializationDeserialization() throws IOException {
        TypeSerializer<QuantileSummary> serializer =
                TypeExtractor.createTypeInfo(QuantileSummary.class)
                        .createSerializer(new ExecutionConfig());
        assertSame(PojoSerializer.class, serializer.getClass());

        PipedInputStream pipedInput = new PipedInputStream(1024 * 1024);
        DataInputViewStreamWrapper reader = new DataInputViewStreamWrapper(pipedInput);
        DataOutputViewStreamWrapper writer =
                new DataOutputViewStreamWrapper(new PipedOutputStream(pipedInput));

        List<QuantileSummary.StatsTuple> sampled =
                Arrays.asList(
                        new QuantileSummary.StatsTuple(10.0, 1L, 1L),
                        new QuantileSummary.StatsTuple(20.0, 1L, 2L),
                        new QuantileSummary.StatsTuple(30.0, 1L, 3L));
        QuantileSummary summary = new QuantileSummary(0.1, 100, sampled, 3, true);
        summary.insert(5.0);

        serializer.serialize(summary, writer);
        writer.write(reader, writer.size());

        QuantileSummary deserialized = serializer.deserialize(reader);
        assertEquals(0.1, deserialized.getRelativeError(), EPS);
        assertEquals(100, deserialized.getCompressThreshold());
        assertEquals(3, deserialized.getSampled().size());
        assertEquals(5.0, deserialized.getHeadBuffer().get(0), EPS);
        assertEquals(3, deserialized.getCount());
        assertFalse(deserialized.isCompressed());
    }
}
