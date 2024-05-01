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

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;

import java.util.Objects;

/**
 * TypeInformation for {@link Long2ObjectOpenHashMap}.
 *
 * @param <T> The type of elements in the Long2ObjectOpenHashMap.
 */
public class Long2ObjectOpenHashMapTypeInfo<T> extends TypeInformation<Long2ObjectOpenHashMap<T>> {

    private final TypeInformation<T> elementTypeInfo;

    public Long2ObjectOpenHashMapTypeInfo(TypeInformation<T> elementTypeInfo) {
        this.elementTypeInfo = elementTypeInfo;
    }

    public TypeInformation<T> getElementTypeInfo() {
        return elementTypeInfo;
    }

    @Override
    public boolean isBasicType() {
        return false;
    }

    @Override
    public boolean isTupleType() {
        return false;
    }

    @Override
    public int getArity() {
        return 1;
    }

    @Override
    public int getTotalFields() {
        return 1;
    }

    @Override
    public Class<Long2ObjectOpenHashMap<T>> getTypeClass() {
        return (Class) Long2ObjectOpenHashMap.class;
    }

    @Override
    public boolean isKeyType() {
        return false;
    }

    @Override
    public TypeSerializer<Long2ObjectOpenHashMap<T>> createSerializer(ExecutionConfig config) {
        return new Long2ObjectOpenHashMapSerializer<>(elementTypeInfo.createSerializer(config));
    }

    @Override
    public String toString() {
        return "Long2ObjectOpenHashMap Type";
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }

        Long2ObjectOpenHashMapTypeInfo<T> that = (Long2ObjectOpenHashMapTypeInfo<T>) obj;
        return Objects.equals(elementTypeInfo, that.elementTypeInfo);
    }

    @Override
    public int hashCode() {
        return Objects.hash(elementTypeInfo != null ? elementTypeInfo.hashCode() : 0);
    }

    @Override
    public boolean canEqual(Object obj) {
        return obj instanceof Long2ObjectOpenHashMapTypeInfo;
    }
}
