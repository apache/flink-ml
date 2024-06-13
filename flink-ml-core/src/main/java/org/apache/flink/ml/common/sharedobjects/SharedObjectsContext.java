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

/**
 * Context for shared objects. Every operator implementing {@link
 * AbstractSharedObjectsStreamOperator} will get an instance of this context set by {@link
 * AbstractSharedObjectsStreamOperator#onSharedObjectsContextSet} in runtime.
 *
 * <p>See {@link ReadRequest} for details about coordination between reads and writes.
 */
@Experimental
public interface SharedObjectsContext {

    /**
     * Reads the value of a shared object.
     *
     * <p>For subclasses of {@link AbstractSharedObjectsOneInputStreamOperator} and {@link
     * AbstractSharedObjectsTwoInputStreamOperator}, this method is guaranteed to return non-null
     * values immediately.
     *
     * @param request A read request of a shared object.
     * @return The value of the shared object.
     * @param <T> The type of the shared object.
     */
    <T> T read(ReadRequest<T> request);

    /**
     * Writes a new value to the shared object.
     *
     * @param descriptor The shared object descriptor.
     * @param value The value to be set.
     * @param <T> The type of the shared object.
     */
    <T> void write(Descriptor<T> descriptor, T value);

    /**
     * Renew the shared object with current step.
     *
     * <p>For subclasses of {@link AbstractSharedObjectsOneInputStreamOperator} and {@link
     * AbstractSharedObjectsTwoInputStreamOperator}, this method is guaranteed to return
     * immediately.
     *
     * @param descriptor The shared object descriptor.
     * @param <T> The type of the shared object.
     */
    <T> void renew(Descriptor<T> descriptor);
}
