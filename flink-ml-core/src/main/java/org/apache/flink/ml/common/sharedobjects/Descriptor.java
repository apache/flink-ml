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

import javax.annotation.Nullable;

import java.io.Serializable;

/**
 * Descriptor for a shared object.
 *
 * <p>A shared object can have a non-null initial value, or have no initial values. If a non-null
 * initial value provided, it is set with an initial write-step (See {@link ReadRequest}).
 *
 * @param <T> The type of the shared object.
 */
@Experimental
public class Descriptor<T> implements Serializable {

    /** Name of the shared object. */
    public final String name;

    /** Type serializer. */
    public final TypeSerializer<T> serializer;

    /** Initialize value. */
    public final @Nullable T initVal;

    private Descriptor(String name, TypeSerializer<T> serializer, T initVal) {
        this.name = name;
        this.serializer = serializer;
        this.initVal = initVal;
    }

    public static <T> Descriptor<T> of(String name, TypeSerializer<T> serializer, T initVal) {
        Preconditions.checkNotNull(
                initVal, "Cannot use `null` as the initial value of a shared object.");
        return new Descriptor<>(name, serializer, initVal);
    }

    public static <T> Descriptor<T> of(String name, TypeSerializer<T> serializer) {
        return new Descriptor<>(name, serializer, null);
    }

    /**
     * Creates a read request which always reads this shared object with same read-step as the
     * operator step.
     *
     * @return A read request.
     */
    public ReadRequest<T> sameStep() {
        return new ReadRequest<>(this, ReadRequest.OFFSET.SAME);
    }

    /**
     * Creates a read request which always reads this shared object with the read-step be the
     * previous item of the operator step.
     *
     * @return A read request.
     */
    public ReadRequest<T> prevStep() {
        return new ReadRequest<>(this, ReadRequest.OFFSET.PREV);
    }

    /**
     * Creates a read request which always reads this shared object with the read-step be the next
     * item of the operator step.
     *
     * @return A read request.
     */
    public ReadRequest<T> nextStep() {
        return new ReadRequest<>(this, ReadRequest.OFFSET.NEXT);
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
        Descriptor<?> that = (Descriptor<?>) o;
        return name.equals(that.name);
    }

    @Override
    public String toString() {
        return String.format(
                "Descriptor{name='%s', serializer=%s, initVal=%s}", name, serializer, initVal);
    }
}
