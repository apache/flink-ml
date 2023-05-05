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

import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.BiConsumerWithException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Default implementation of {@link SharedObjectsContext}.
 *
 * <p>It initializes readers and writers according to the owner map when the subtask starts and
 * clean internal states when the subtask finishes. It also handles `initializeState` and
 * `snapshotState` automatically.
 */
@SuppressWarnings("rawtypes")
class SharedObjectsContextImpl implements SharedObjectsContext, Serializable {
    private final PoolID poolID;
    private final Map<ItemDescriptor, SharedObjectsPools.Writer> writers = new HashMap<>();
    private final Map<ItemDescriptor, SharedObjectsPools.Reader> readers = new HashMap<>();
    private Map<ItemDescriptor<?>, String> ownerMap;

    public SharedObjectsContextImpl() {
        this.poolID = new PoolID();
    }

    void setOwnerMap(Map<ItemDescriptor<?>, String> ownerMap) {
        this.ownerMap = ownerMap;
    }

    @Override
    public void invoke(BiConsumerWithException<SharedItemGetter, SharedItemSetter, Exception> func)
            throws Exception {
        func.accept(this::getSharedItem, this::setSharedItem);
    }

    private <T> T getSharedItem(ItemDescriptor<T> key) {
        //noinspection unchecked
        SharedObjectsPools.Reader<T> reader = readers.get(key);
        Preconditions.checkState(
                null != reader,
                String.format(
                        "The operator requested to read a shared item %s not owned by itself.",
                        key));
        return reader.get();
    }

    private <T> void setSharedItem(ItemDescriptor<T> key, T value) {
        //noinspection unchecked
        SharedObjectsPools.Writer<T> writer = writers.get(key);
        Preconditions.checkState(
                null != writer,
                String.format(
                        "The operator requested to read a shared item %s not owned by itself.",
                        key));
        writer.set(value);
    }

    void initializeState(
            StreamOperator<?> operator,
            StreamingRuntimeContext runtimeContext,
            StateInitializationContext context) {
        Preconditions.checkArgument(
                operator instanceof SharedObjectsStreamOperator
                        && operator instanceof AbstractStreamOperator);
        String ownerId = ((SharedObjectsStreamOperator) operator).getSharedObjectsAccessorID();
        int subtaskId = runtimeContext.getIndexOfThisSubtask();
        for (Map.Entry<ItemDescriptor<?>, String> entry : ownerMap.entrySet()) {
            ItemDescriptor descriptor = entry.getKey();
            if (ownerId.equals(entry.getValue())) {
                writers.put(
                        descriptor,
                        SharedObjectsPools.getWriter(
                                poolID,
                                subtaskId,
                                descriptor,
                                ownerId,
                                operator.getOperatorID(),
                                ((AbstractStreamOperator<?>) operator).getContainingTask(),
                                runtimeContext,
                                context));
            }
            readers.put(descriptor, SharedObjectsPools.getReader(poolID, subtaskId, descriptor));
        }
    }

    void snapshotState(StateSnapshotContext context) throws Exception {
        for (SharedObjectsPools.Writer<?> writer : writers.values()) {
            writer.snapshotState(context);
        }
    }

    void clear() {
        for (SharedObjectsPools.Writer writer : writers.values()) {
            writer.remove();
        }
        for (SharedObjectsPools.Reader reader : readers.values()) {
            reader.remove();
        }
        writers.clear();
        readers.clear();
    }
}
