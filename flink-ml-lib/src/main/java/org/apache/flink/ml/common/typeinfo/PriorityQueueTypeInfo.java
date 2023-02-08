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
import org.apache.flink.api.common.typeutils.TypeSerializer;

import java.util.Comparator;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * TypeInformation for {@link java.util.PriorityQueue}.
 *
 * @param <T> The type of inner objects.
 */
public class PriorityQueueTypeInfo<T> extends TypeInformation<PriorityQueue<T>> {

    private final Comparator<? super T> comparator;

    private final TypeInformation<T> innerTypeInfo;

    public PriorityQueueTypeInfo(
            Comparator<? super T> comparator, TypeInformation<T> innerTypeInfo) {
        this.comparator = comparator;
        this.innerTypeInfo = innerTypeInfo;
    }

    public TypeInformation<T> getInnerTypeInfo() {
        return innerTypeInfo;
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
    public Class<PriorityQueue<T>> getTypeClass() {
        return (Class) PriorityQueue.class;
    }

    @Override
    public boolean isKeyType() {
        return false;
    }

    @Override
    public TypeSerializer<PriorityQueue<T>> createSerializer(ExecutionConfig config) {
        return new PriorityQueueSerializer<>(comparator, innerTypeInfo.createSerializer(config));
    }

    @Override
    public String toString() {
        return "PriorityQueue Type";
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }

        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }

        PriorityQueueTypeInfo<T> that = (PriorityQueueTypeInfo<T>) obj;
        return Objects.equals(comparator, that.comparator)
                && Objects.equals(innerTypeInfo, that.innerTypeInfo);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                comparator != null ? comparator.hashCode() : 0,
                innerTypeInfo != null ? innerTypeInfo.hashCode() : 0);
    }

    @Override
    public boolean canEqual(Object obj) {
        return obj instanceof PriorityQueueTypeInfo;
    }
}
