/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.iteration.broadcast;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.runtime.io.network.api.writer.RecordWriter;
import org.apache.flink.runtime.plugable.SerializationDelegate;
import org.apache.flink.streaming.runtime.streamrecord.StreamElement;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import java.io.IOException;

/** The broadcast output corresponding to a record writer output. */
public class RecordWriterBroadcastOutput<OUT> implements BroadcastOutput<OUT> {
    private final RecordWriter<SerializationDelegate<StreamElement>> recordWriter;
    private final SerializationDelegate<StreamElement> serializationDelegate;

    public RecordWriterBroadcastOutput(
            RecordWriter<SerializationDelegate<StreamElement>> recordWriter,
            TypeSerializer<StreamElement> typeSerializer) {

        this.recordWriter = recordWriter;
        this.serializationDelegate = new SerializationDelegate<>(typeSerializer);
    }

    @Override
    public void broadcastEmit(StreamRecord<OUT> record) throws IOException {
        serializationDelegate.setInstance(record);
        recordWriter.broadcastEmit(serializationDelegate);
        if (isIterationEpochWatermark(record)) {
            recordWriter.flushAll();
        }
    }

    private static <T> boolean isIterationEpochWatermark(StreamRecord<T> record) {
        if (!(record.getValue() instanceof IterationRecord)) {
            return false;
        }
        IterationRecord<?> iterationRecord = (IterationRecord<?>) record.getValue();
        return iterationRecord.getType().equals(IterationRecord.Type.EPOCH_WATERMARK);
    }
}
