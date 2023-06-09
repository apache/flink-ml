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

package org.apache.flink.iteration.proxy;

import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;
import org.apache.flink.util.OutputTag;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/** Proxy output to provide to the wrapped operator. */
public class ProxyOutput<T> implements Output<StreamRecord<T>> {

    private final Output<StreamRecord<IterationRecord<T>>> output;

    private final StreamRecord<IterationRecord<T>> reuseRecord;

    private final Map<String, SideOutputCache> sideOutputCaches = new HashMap<>();

    private Integer contextRound;

    public ProxyOutput(Output<StreamRecord<IterationRecord<T>>> output) {
        this.output = Objects.requireNonNull(output);
        this.reuseRecord = new StreamRecord<>(IterationRecord.newRecord(null, 0));
    }

    public void setContextRound(Integer contextRound) {
        this.contextRound = contextRound;
    }

    @Override
    public void emitWatermark(Watermark mark) {
        // For now, we only supports the MAX_WATERMARK separately for each operator.
    }

    @Override
    public void emitWatermarkStatus(WatermarkStatus watermarkStatus) {
        output.emitWatermarkStatus(watermarkStatus);
    }

    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public <X> void collect(OutputTag<X> outputTag, StreamRecord<X> record) {
        SideOutputCache sideOutputCache =
                sideOutputCaches.computeIfAbsent(
                        outputTag.getId(),
                        (ignored) ->
                                new SideOutputCache(
                                        new OutputTag<IterationRecord<?>>(
                                                outputTag.getId(),
                                                new IterationRecordTypeInfo(
                                                        outputTag.getTypeInfo())),
                                        new StreamRecord<>(IterationRecord.newRecord(null, 0))));
        sideOutputCache.cachedRecord.replace(
                IterationRecord.newRecord(record.getValue(), contextRound), record.getTimestamp());
        output.collect(sideOutputCache.tag, sideOutputCache.cachedRecord);
    }

    @Override
    public void emitLatencyMarker(LatencyMarker latencyMarker) {
        output.emitLatencyMarker(latencyMarker);
    }

    @Override
    public void collect(StreamRecord<T> record) {
        reuseRecord.getValue().setValue(record.getValue());
        reuseRecord.getValue().setEpoch(contextRound);
        reuseRecord.setTimestamp(record.getTimestamp());
        output.collect(reuseRecord);
    }

    @Override
    public void close() {
        output.close();
    }

    private static class SideOutputCache {
        final OutputTag<IterationRecord<?>> tag;

        final StreamRecord<IterationRecord<?>> cachedRecord;

        public SideOutputCache(
                OutputTag<IterationRecord<?>> tag, StreamRecord<IterationRecord<?>> cachedRecord) {
            this.tag = tag;
            this.cachedRecord = cachedRecord;
        }
    }
}
