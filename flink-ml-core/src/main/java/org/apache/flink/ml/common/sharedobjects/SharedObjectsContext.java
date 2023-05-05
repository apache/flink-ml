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
import org.apache.flink.util.function.BiConsumerWithException;

/**
 * Context for shared objects. Every operator implementing {@link SharedObjectsStreamOperator} will
 * get an instance of this context set by {@link
 * SharedObjectsStreamOperator#onSharedObjectsContextSet} in runtime. User-defined logic can be
 * invoked through {@link #invoke} with the access to shared items.
 */
@Experimental
public interface SharedObjectsContext {

    /**
     * Invoke user defined function with provided getters/setters of the shared objects.
     *
     * @param func User defined function where share items can be accessed through getters/setters.
     * @throws Exception Possible exception.
     */
    void invoke(BiConsumerWithException<SharedItemGetter, SharedItemSetter, Exception> func)
            throws Exception;

    /** Interface of shared item getter. */
    @FunctionalInterface
    interface SharedItemGetter {
        <T> T get(ItemDescriptor<T> key);
    }

    /** Interface of shared item writer. */
    @FunctionalInterface
    interface SharedItemSetter {
        <T> void set(ItemDescriptor<T> key, T value);
    }
}
