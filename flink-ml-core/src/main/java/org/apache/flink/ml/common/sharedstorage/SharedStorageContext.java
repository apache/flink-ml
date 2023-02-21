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
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorStateHandler;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.util.function.BiConsumerWithException;

/**
 * Context for shared storage. Every operator implementing {@link SharedStorageStreamOperator} will
 * have an instance of this context set by {@link
 * SharedStorageStreamOperator#onSharedStorageContextSet} in runtime. User defined logic can be
 * invoked through {@link #invoke} with the access to shared items.
 *
 * <p>NOTE: The corresponding operator must explicitly invoke
 *
 * <ul>
 *   <li>{@link #initializeState} to initialize this context and possibly restore data items owned
 *       by itself in {@link StreamOperatorStateHandler.CheckpointedStreamOperator#initializeState};
 *   <li>{@link #snapshotState} in order to save data items owned by itself in {@link
 *       StreamOperatorStateHandler.CheckpointedStreamOperator#snapshotState};
 *   <li>{@link #clear()} in order to clear all data items owned by itself in {@link
 *       StreamOperator#close}.
 * </ul>
 */
@Experimental
public interface SharedStorageContext {

    /**
     * Invoke user defined function with provided getters/setters of the shared storage.
     *
     * @param func User defined function where share items can be accessed through getters/setters.
     * @throws Exception Possible exception.
     */
    void invoke(BiConsumerWithException<SharedItemGetter, SharedItemSetter, Exception> func)
            throws Exception;

    /** Initializes shared storage context and restores of shared items owned by this operator. */
    <T extends AbstractStreamOperator<?> & SharedStorageStreamOperator> void initializeState(
            T operator, StreamingRuntimeContext runtimeContext, StateInitializationContext context);

    /** Save shared items owned by this operator. */
    void snapshotState(StateSnapshotContext context) throws Exception;

    /** Clear all internal states. */
    void clear();

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
