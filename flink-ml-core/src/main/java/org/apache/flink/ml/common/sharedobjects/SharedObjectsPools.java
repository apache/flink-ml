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
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.runtime.TupleSerializer;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.AbstractID;
import org.apache.flink.util.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;

/**
 * Stores all shared objects and coordinates their reads and writes.
 *
 * <p>Every shared object is identified by a tuple of (Pool ID, subtask ID, name). Their reads and
 * writes are coordinated through the read- and write-steps.
 */
class SharedObjectsPools {

    private static final Logger LOG = LoggerFactory.getLogger(SharedObjectsPools.class);

    /** Stores values and corresponding write-steps of all shared objects. */
    private static final Map<Tuple3<PoolID, Integer, String>, Tuple2<Integer, Object>> values =
            new ConcurrentHashMap<>();

    /**
     * Stores waiting read requests of all shared objects, including read-steps and count-down
     * latches for notification when shared objects are ready.
     */
    private static final Map<Tuple3<PoolID, Integer, String>, List<Tuple2<Integer, CountDownLatch>>>
            waitQueues = new ConcurrentHashMap<>();

    /**
     * Stores owners of all shared objects, where the owner is identified by the accessor ID
     * obtained from {@link AbstractSharedObjectsStreamOperator#getAccessorID()}.
     */
    private static final Map<Tuple3<PoolID, Integer, String>, String> owners =
            new ConcurrentHashMap<>();

    /**
     * Stores number of references of all shared objects. Every {@link Reader} and {@link Writer}
     * counts. A shared object is removed from the pool when its number of references decreased to
     * 0.
     */
    private static final ConcurrentHashMap<Tuple3<PoolID, Integer, String>, Integer> numRefs =
            new ConcurrentHashMap<>();

    private static void incRef(Tuple3<PoolID, Integer, String> objId) {
        numRefs.compute(objId, (k, oldV) -> null == oldV ? 1 : oldV + 1);
    }

    private static void decRef(Tuple3<PoolID, Integer, String> objId) {
        int num = numRefs.compute(objId, (k, oldV) -> oldV - 1);
        if (num == 0) {
            values.remove(objId);
            waitQueues.remove(objId);
            owners.remove(objId);
            numRefs.remove(objId);
        }
    }

    /** Gets a {@link Reader} of a shared object. */
    static <T> Reader<T> getReader(PoolID poolID, int subtaskId, Descriptor<T> descriptor) {
        Tuple3<PoolID, Integer, String> objId = Tuple3.of(poolID, subtaskId, descriptor.name);
        Reader<T> reader = new Reader<>(objId);
        incRef(objId);
        return reader;
    }

    /** Gets a {@link Writer} of a shared object. */
    static <T> Writer<T> getWriter(
            PoolID poolId,
            int subtaskId,
            Descriptor<T> descriptor,
            String ownerId,
            OperatorID operatorID,
            StreamTask<?, ?> containingTask,
            StreamingRuntimeContext runtimeContext,
            StateInitializationContext stateInitializationContext,
            int step) {
        Tuple3<PoolID, Integer, String> objId = Tuple3.of(poolId, subtaskId, descriptor.name);
        String lastOwner = owners.putIfAbsent(objId, ownerId);
        if (null != lastOwner) {
            throw new IllegalStateException(
                    String.format(
                            "The shared object (%s, %s, %s) already has a writer %s.",
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
        incRef(objId);
        if (null != descriptor.initVal) {
            writer.set(descriptor.initVal, step);
        }
        return writer;
    }

    /**
     * Reader of a shared object.
     *
     * @param <T> The type of the shared object.
     */
    static class Reader<T> {
        protected final Tuple3<PoolID, Integer, String> objId;

        Reader(Tuple3<PoolID, Integer, String> objId) {
            this.objId = objId;
        }

        /**
         * Gets the value with given read-step. There are 3 cases:
         *
         * <ol>
         *   <li>The read-step is equal to the write-step: returns the value immediately.
         *   <li>The read-step is larger than the write-step, or there is no values written yet:
         *       waits until the value with same write-step set if `wait` is true, or returns null
         *       otherwise.
         *   <li>The read-step is smaller than the write-step: throws an exception as it is illegal.
         * </ol>
         *
         * @param readStep The read-step.
         * @param wait Whether to wait until the value with same write-step presents.
         * @return The value or null. A return value of null means the corresponding value if not
         *     presented. If `wait` is true, the return value of this function is guaranteed to be a
         *     non-null value if it returns.
         * @throws InterruptedException Interrupted when waiting.
         */
        T get(int readStep, boolean wait) throws InterruptedException {
            //noinspection unchecked
            Tuple2<Integer, T> stepV = (Tuple2<Integer, T>) values.get(objId);
            if (null != stepV) {
                int writeStep = stepV.f0;
                LOG.debug("Get {} with read-step {}, write-step is {}", objId, readStep, writeStep);
                Preconditions.checkState(
                        writeStep <= readStep,
                        String.format(
                                "Current write-step %d of %s is larger than read-step %d, which is illegal.",
                                writeStep, objId, readStep));
                if (readStep == stepV.f0) {
                    return stepV.f1;
                }
            }
            if (!wait) {
                return null;
            }
            CountDownLatch latch = new CountDownLatch(1);
            synchronized (waitQueues) {
                if (!waitQueues.containsKey(objId)) {
                    waitQueues.put(objId, new ArrayList<>());
                }
                List<Tuple2<Integer, CountDownLatch>> q = waitQueues.get(objId);
                q.add(Tuple2.of(readStep, latch));
            }
            latch.await();
            //noinspection unchecked
            stepV = (Tuple2<Integer, T>) values.get(objId);
            Preconditions.checkState(stepV.f0 == readStep);
            return stepV.f1;
        }

        void remove() {
            decRef(objId);
        }
    }

    /**
     * Writer of a shared object.
     *
     * @param <T> The type of the shared object.
     */
    static class Writer<T> extends Reader<T> {
        private final String ownerId;
        private final ListStateWithCache<Tuple2<Integer, T>> cache;
        private boolean isDirty;

        Writer(
                Tuple3<PoolID, Integer, String> objId,
                String ownerId,
                TypeSerializer<T> serializer,
                StreamTask<?, ?> containingTask,
                StreamingRuntimeContext runtimeContext,
                StateInitializationContext stateInitializationContext,
                OperatorID operatorID) {
            super(objId);
            this.ownerId = ownerId;
            try {
                //noinspection unchecked
                cache =
                        new ListStateWithCache<>(
                                new TupleSerializer<>(
                                        (Class<Tuple2<Integer, T>>) (Class<?>) Tuple2.class,
                                        new TypeSerializer[] {IntSerializer.INSTANCE, serializer}),
                                containingTask,
                                runtimeContext,
                                stateInitializationContext,
                                operatorID);
                Iterator<Tuple2<Integer, T>> iterator = cache.get().iterator();
                if (iterator.hasNext()) {
                    Tuple2<Integer, T> stepV = iterator.next();
                    ensureOwner();
                    //noinspection unchecked
                    values.put(objId, (Tuple2<Integer, Object>) stepV);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            isDirty = false;
        }

        private void ensureOwner() {
            Preconditions.checkState(owners.get(objId).equals(ownerId));
        }

        /**
         * Sets the value with given write-step. If there are read requests waiting for the value of
         * exact the same write-step, they are notified.
         *
         * @param value The value.
         * @param writeStep The write-step.
         */
        void set(T value, int writeStep) {
            ensureOwner();
            values.put(objId, Tuple2.of(writeStep, value));
            LOG.debug("Set {} with write-step {}", objId, writeStep);
            isDirty = true;
            synchronized (waitQueues) {
                if (!waitQueues.containsKey(objId)) {
                    waitQueues.put(objId, new ArrayList<>());
                }
                List<Tuple2<Integer, CountDownLatch>> q = waitQueues.get(objId);
                ListIterator<Tuple2<Integer, CountDownLatch>> iter = q.listIterator();
                while (iter.hasNext()) {
                    Tuple2<Integer, CountDownLatch> next = iter.next();
                    if (writeStep == next.f0) {
                        iter.remove();
                        next.f1.countDown();
                    }
                }
            }
        }

        @Override
        void remove() {
            ensureOwner();
            super.remove();
            cache.clear();
        }

        void snapshotState(StateSnapshotContext context) throws Exception {
            if (isDirty) {
                //noinspection unchecked
                cache.update(Collections.singletonList((Tuple2<Integer, T>) values.get(objId)));
                isDirty = false;
            }
            cache.snapshotState(context);
        }
    }

    /** ID of a pool for shared objects. */
    static class PoolID extends AbstractID {
        private static final long serialVersionUID = 1L;

        public PoolID(byte[] bytes) {
            super(bytes);
        }

        public PoolID() {}
    }
}
