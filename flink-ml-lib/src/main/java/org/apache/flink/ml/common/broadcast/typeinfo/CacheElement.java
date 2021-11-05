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

package org.apache.flink.ml.common.broadcast.typeinfo;

/**
 * The wrapper class for possible cached elements used in {@link
 * org.apache.flink.ml.common.broadcast.operator.AbstractBroadcastWrapperOperator}. It could be
 * either {@link org.apache.flink.streaming.api.watermark.Watermark}, {@link
 * org.apache.flink.streaming.runtime.streamrecord.StreamRecord}, etc.
 *
 * @param <T> the record type.
 */
public class CacheElement<T> {
    private T record;
    private long watermark;
    private Type type;

    public CacheElement(T record, long watermark, Type type) {
        this.record = record;
        this.watermark = watermark;
        this.type = type;
    }

    public static <T> CacheElement<T> newRecord(T record) {
        return new CacheElement<>(record, -1, Type.RECORD);
    }

    public static <T> CacheElement<T> newWatermark(long watermark) {
        return new CacheElement<>(null, watermark, Type.WATERMARK);
    }

    public T getRecord() {
        return record;
    }

    public void setRecord(T record) {
        this.record = record;
    }

    public long getWatermark() {
        return watermark;
    }

    public void setWatermark(long watermark) {
        this.watermark = watermark;
    }

    public Type getType() {
        return type;
    }

    public void setType(Type type) {
        this.type = type;
    }

    /** The type of cached elements. */
    public enum Type {
        /** record type. */
        RECORD,
        /** watermark type. */
        WATERMARK,
    }
}
