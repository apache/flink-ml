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

/**
 * Stores and manages all shared objects. Every shared object is identified by a tuple of (Pool ID,
 * subtask ID, name). Every call of {@link SharedObjectsUtils#withSharedObjects} generated a
 * different {@link PoolID}, so that they do not interfere with each other.
 */
class SharedObjectsPools {

    // Stores values of all shared objects.
    private static final Map<Tuple3<PoolID, Integer, String>, Object> values =
            new ConcurrentHashMap<>();

    /**
     * Stores owners of all shared objects, where the owner is identified by the accessor ID
     * obtained from {@link SharedObjectsStreamOperator#getSharedObjectsAccessorID()}.
     */
    private static final Map<Tuple3<PoolID, Integer, String>, String> owners =
            new ConcurrentHashMap<>();

    // Stores number of references of all shared objects. A shared object is removed when its number
    // of references decreased to 0.
    private static final ConcurrentHashMap<Tuple3<PoolID, Integer, String>, Integer> numRefs =
            new ConcurrentHashMap<>();

    @SuppressWarnings("UnusedReturnValue")
    static int incRef(Tuple3<PoolID, Integer, String> itemId) {
        return numRefs.compute(itemId, (k, oldV) -> null == oldV ? 1 : oldV + 1);
    }

    @SuppressWarnings("UnusedReturnValue")
    static int decRef(Tuple3<PoolID, Integer, String> itemId) {
        int num = numRefs.compute(itemId, (k, oldV) -> oldV - 1);
        if (num == 0) {
            values.remove(itemId);
            owners.remove(itemId);
            numRefs.remove(itemId);
        }
        return num;
    }

    /** Gets a {@link Reader} of a shared object. */
    static <T> Reader<T> getReader(PoolID poolID, int subtaskId, ItemDescriptor<T> descriptor) {
        Tuple3<PoolID, Integer, String> itemId = Tuple3.of(poolID, subtaskId, descriptor.name);
        Reader<T> reader = new Reader<>(itemId);
        incRef(itemId);
        return reader;
    }

    /** Gets a {@link Writer} of a shared object. */
    static <T> Writer<T> getWriter(
            PoolID poolId,
            int subtaskId,
            ItemDescriptor<T> descriptor,
            String ownerId,
            OperatorID operatorID,
            StreamTask<?, ?> containingTask,
            StreamingRuntimeContext runtimeContext,
            StateInitializationContext stateInitializationContext) {
        Tuple3<PoolID, Integer, String> objId = Tuple3.of(poolId, subtaskId, descriptor.name);
        String lastOwner = owners.putIfAbsent(objId, ownerId);
        if (null != lastOwner) {
            throw new IllegalStateException(
                    String.format(
                            "The shared item (%s, %s, %s) already has a writer %s.",
                            poolId, subtaskId, descriptor.name, ownerId));
        }
        Writer<T> writer =
                new Writer<>(
                        objId,
                        ownerId,
                        descriptor.serializer,
                        containingTask,
                        runtimeContext,
                        stateInitializationContext,
                        operatorID);
        writer.set(descriptor.initVal);
        incRef(objId);
        return writer;
    }

    static class Reader<T> {
        protected final Tuple3<PoolID, Integer, String> objId;

        Reader(Tuple3<PoolID, Integer, String> objId) {
            this.objId = objId;
        }

        T get() {
            // It is possible that the `get` request of an item is triggered earlier than its
            // initialization. In this case, we wait for a while.
            long waitTime = 10;
            do {
                //noinspection unchecked
                T value = (T) values.get(objId);
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
                    String.format(
                            "Failed to get value of %s after waiting %d ms.", objId, waitTime));
        }

        void remove() {
            decRef(objId);
        }
    }

    static class Writer<T> extends Reader<T> {
        private final String ownerId;
        private final ListStateWithCache<T> cache;
        private boolean isDirty;

        Writer(
                Tuple3<PoolID, Integer, String> itemId,
                String ownerId,
                TypeSerializer<T> serializer,
                StreamTask<?, ?> containingTask,
                StreamingRuntimeContext runtimeContext,
                StateInitializationContext stateInitializationContext,
                OperatorID operatorID) {
            super(itemId);
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
                    values.put(itemId, value);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            isDirty = false;
        }

        private void ensureOwner() {
            // Double-checks the owner, because a writer may call this method after the key removed
            // and re-added by other operators.
            Preconditions.checkState(owners.get(objId).equals(ownerId));
        }

        void set(T value) {
            ensureOwner();
            values.put(objId, value);
            isDirty = true;
        }

        @Override
        void remove() {
            ensureOwner();
            super.remove();
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
