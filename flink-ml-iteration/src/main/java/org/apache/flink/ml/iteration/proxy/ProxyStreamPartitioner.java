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

package org.apache.flink.ml.iteration.proxy;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.ml.iteration.IterationRecord;
import org.apache.flink.ml.iteration.typeinfo.IterationRecordSerializer;
import org.apache.flink.ml.iteration.utils.ReflectionUtils;
import org.apache.flink.runtime.io.network.api.writer.SubtaskStateMapper;
import org.apache.flink.runtime.plugable.SerializationDelegate;
import org.apache.flink.streaming.runtime.partitioner.StreamPartitioner;
import org.apache.flink.streaming.runtime.streamrecord.StreamElementSerializer;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import java.util.Objects;

/** Proxy stream partitioner for the wrapped one. */
public class ProxyStreamPartitioner<T> extends StreamPartitioner<IterationRecord<T>> {

    private final StreamPartitioner<T> wrappedStreamPartitioner;

    private transient SerializationDelegate<StreamRecord<T>> reuseDelegate;

    private transient StreamRecord<T> reuseRecord;

    public ProxyStreamPartitioner(StreamPartitioner<T> wrappedStreamPartitioner) {
        this.wrappedStreamPartitioner = Objects.requireNonNull(wrappedStreamPartitioner);
    }

    @Override
    public StreamPartitioner<IterationRecord<T>> copy() {
        return new ProxyStreamPartitioner<>(wrappedStreamPartitioner.copy());
    }

    @Override
    public SubtaskStateMapper getDownstreamSubtaskStateMapper() {
        return wrappedStreamPartitioner.getDownstreamSubtaskStateMapper();
    }

    @Override
    public boolean isPointwise() {
        return wrappedStreamPartitioner.isPointwise();
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public int selectChannel(SerializationDelegate<StreamRecord<IterationRecord<T>>> record) {
        if (reuseDelegate != null) {
            reuseDelegate.setInstance(
                    reuseRecord.replace(
                            record.getInstance().getValue().getValue(),
                            record.getInstance().getTimestamp()));
            return wrappedStreamPartitioner.selectChannel(reuseDelegate);
        } else {
            reuseRecord = new StreamRecord<>(null, 0);

            StreamElementSerializer<IterationRecord<T>> streamElementSerializer =
                    ReflectionUtils.getFieldValue(
                            record, SerializationDelegate.class, "serializer");
            IterationRecordSerializer<T> iterationRecordSerializer =
                    (IterationRecordSerializer<T>)
                            streamElementSerializer.getContainedTypeSerializer();
            reuseDelegate =
                    new SerializationDelegate<>(
                            (TypeSerializer)
                                    new StreamElementSerializer<>(
                                            iterationRecordSerializer
                                                    .getInnerSerializer()
                                                    .duplicate()));

            return selectChannel(record);
        }
    }
}
