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

package org.apache.flink.ml.common.ps.typeinfo;

import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;

import java.io.IOException;
import java.util.Map;

/** TypeSerializer for {@link Long2DoubleOpenHashMap}. */
public class Long2DoubleOpenHashMapSerializer extends TypeSerializer<Long2DoubleOpenHashMap> {

    public static final Long2DoubleOpenHashMapSerializer INSTANCE =
            new Long2DoubleOpenHashMapSerializer();

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<Long2DoubleOpenHashMap> duplicate() {
        return INSTANCE;
    }

    @Override
    public Long2DoubleOpenHashMap createInstance() {
        return new Long2DoubleOpenHashMap();
    }

    @Override
    public Long2DoubleOpenHashMap copy(Long2DoubleOpenHashMap from) {
        return new Long2DoubleOpenHashMap(from);
    }

    @Override
    public Long2DoubleOpenHashMap copy(Long2DoubleOpenHashMap from, Long2DoubleOpenHashMap reuse) {
        return new Long2DoubleOpenHashMap(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Long2DoubleOpenHashMap map, DataOutputView target) throws IOException {
        target.writeInt(map.size());
        for (Map.Entry<Long, Double> entry : map.entrySet()) {
            target.writeLong(entry.getKey());
            target.writeDouble(entry.getValue());
        }
    }

    @Override
    public Long2DoubleOpenHashMap deserialize(DataInputView source) throws IOException {
        int numEntries = source.readInt();
        Long2DoubleOpenHashMap map = new Long2DoubleOpenHashMap(numEntries);
        for (int i = 0; i < numEntries; i++) {
            long k = source.readLong();
            double v = source.readDouble();
            map.put(k, v);
        }
        return map;
    }

    @Override
    public Long2DoubleOpenHashMap deserialize(Long2DoubleOpenHashMap reuse, DataInputView source)
            throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        int numEntries = source.readInt();
        target.writeInt(numEntries);
        for (int i = 0; i < numEntries; ++i) {
            target.writeLong(source.readLong());
            target.writeDouble(source.readDouble());
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        return 0;
    }

    @Override
    public TypeSerializerSnapshot<Long2DoubleOpenHashMap> snapshotConfiguration() {
        return new Long2DoubleOpenHashMapSnapshot();
    }

    private static final class Long2DoubleOpenHashMapSnapshot
            extends SimpleTypeSerializerSnapshot<Long2DoubleOpenHashMap> {
        public Long2DoubleOpenHashMapSnapshot() {
            super(() -> Long2DoubleOpenHashMapSerializer.INSTANCE);
        }
    }
}
