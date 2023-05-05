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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.util.Preconditions;

import java.io.Serializable;

/**
 * Descriptor for a shared item.
 *
 * @param <T> The type of the shared item.
 */
@Experimental
public class ItemDescriptor<T> implements Serializable {

    /** Name of the item. */
    public final String name;

    /** Type serializer. */
    public final TypeSerializer<T> serializer;

    /** Initialize value. */
    public final T initVal;

    private ItemDescriptor(String name, TypeSerializer<T> serializer, T initVal) {
        Preconditions.checkNotNull(
                initVal, "Cannot use `null` as the initial value of a shared item.");
        this.name = name;
        this.serializer = serializer;
        this.initVal = initVal;
    }

    public static <T> ItemDescriptor<T> of(String name, TypeSerializer<T> serializer, T initVal) {
        return new ItemDescriptor<>(name, serializer, initVal);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
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
        return name.equals(that.name);
    }

    @Override
    public String toString() {
        return String.format(
                "ItemDescriptor{name='%s', serializer=%s, initVal=%s}", name, serializer, initVal);
    }
}
