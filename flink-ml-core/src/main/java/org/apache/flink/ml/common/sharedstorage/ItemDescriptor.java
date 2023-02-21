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

package org.apache.flink.ml.common.sharedstorage;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.api.common.typeutils.TypeSerializer;

import java.io.Serializable;

/**
 * Descriptor for a shared item.
 *
 * @param <T> The type of the shared item.
 */
@Experimental
public class ItemDescriptor<T> implements Serializable {

    /** Name of the item. */
    public String key;

    /** Type serializer. */
    public TypeSerializer<T> serializer;

    /** Initialize value. */
    public T initVal;

    private ItemDescriptor(String key, TypeSerializer<T> serializer, T initVal) {
        this.key = key;
        this.serializer = serializer;
        this.initVal = initVal;
    }

    public static <T> ItemDescriptor<T> of(String key, TypeSerializer<T> serializer, T initVal) {
        return new ItemDescriptor<>(key, serializer, initVal);
    }

    @Override
    public int hashCode() {
        return key.hashCode();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        ItemDescriptor<?> that = (ItemDescriptor<?>) o;
        return key.equals(that.key);
    }

    @Override
    public String toString() {
        return String.format(
                "ItemDescriptor{key='%s', serializer=%s, initVal=%s}", key, serializer, initVal);
    }
}
