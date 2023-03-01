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

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.Preconditions;

import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A shared storage to support access through subtasks of different operators. */
class SharedStorage {
    private static final Map<Tuple3<StorageID, Integer, String>, Object> m =
            new ConcurrentHashMap<>();

    private static final Map<Tuple3<StorageID, Integer, String>, String> owners =
            new ConcurrentHashMap<>();

    /** Gets a {@link Reader} of shared item identified by (storageID, subtaskId, descriptor). */
    static <T> Reader<T> getReader(
            StorageID storageID, int subtaskId, ItemDescriptor<T> descriptor) {
        return new Reader<>(Tuple3.of(storageID, subtaskId, descriptor.key));
    }

    /** Gets a {@link Writer} of shared item identified by (storageID, subtaskId, key). */
    static <T> Writer<T> getWriter(
            StorageID storageID,
            int subtaskId,
            ItemDescriptor<T> descriptor,
            String ownerId,
            OperatorID operatorID,
            StreamTask<?, ?> containingTask,
            StreamingRuntimeContext runtimeContext,
            StateInitializationContext stateInitializationContext) {
        Tuple3<StorageID, Integer, String> t = Tuple3.of(storageID, subtaskId, descriptor.key);
        String lastOwner = owners.putIfAbsent(t, ownerId);
        if (null != lastOwner) {
            throw new IllegalStateException(
                    String.format(
                            "The shared item (%s, %s, %s) already has a writer %s.",
                            storageID, subtaskId, descriptor.key, ownerId));
        }
        Writer<T> writer =
                new Writer<>(
                        t,
                        ownerId,
                        descriptor.serializer,
                        containingTask,
                        runtimeContext,
                        stateInitializationContext,
                        operatorID);
        writer.set(descriptor.initVal);
        return writer;
    }

    static class Reader<T> {
        protected final Tuple3<StorageID, Integer, String> t;

        Reader(Tuple3<StorageID, Integer, String> t) {
            this.t = t;
        }

        T get() {
            // It is possible that the `get` request of an item is triggered earlier than its
            // initialization. In this case, we wait for a while.
            long waitTime = 10;
            do {
                //noinspection unchecked
                T value = (T) m.get(t);
                if (null != value) {
                    return value;
                }
                try {
                    Thread.sleep(waitTime);
                } catch (InterruptedException e) {
                    break;
                }
                waitTime *= 2;
            } while (waitTime < 10 * 1000);
            throw new IllegalStateException(
                    String.format("Failed to get value of %s after waiting %d ms.", t, waitTime));
        }
    }

    static class Writer<T> extends Reader<T> {
        private final String ownerId;
        private final ListStateWithCache<T> cache;
        private boolean isDirty;

        Writer(
                Tuple3<StorageID, Integer, String> t,
                String ownerId,
                TypeSerializer<T> serializer,
                StreamTask<?, ?> containingTask,
                StreamingRuntimeContext runtimeContext,
                StateInitializationContext stateInitializationContext,
                OperatorID operatorID) {
            super(t);
            this.ownerId = ownerId;
            try {
                cache =
                        new ListStateWithCache<>(
                                serializer,
                                containingTask,
                                runtimeContext,
                                stateInitializationContext,
                                operatorID);
                Iterator<T> iterator = cache.get().iterator();
                if (iterator.hasNext()) {
                    T value = iterator.next();
                    ensureOwner();
                    m.put(t, value);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            isDirty = false;
        }

        private void ensureOwner() {
            // Double-checks the owner, because a writer may call this method after the key removed
            // and re-added by other operators.
            Preconditions.checkState(owners.get(t).equals(ownerId));
        }

        void set(T value) {
            ensureOwner();
            m.put(t, value);
            isDirty = true;
        }

        void remove() {
            ensureOwner();
            m.remove(t);
            owners.remove(t);
            cache.clear();
        }

        void snapshotState(StateSnapshotContext context) throws Exception {
            if (isDirty) {
                cache.update(Collections.singletonList(get()));
                isDirty = false;
            }
            cache.snapshotState(context);
        }
    }
}
