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

package org.apache.flink.ml.common.gbt.datastorage;

import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.util.Preconditions;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A shared storage across subtasks of different operators. */
public class IterationSharedStorage {
    private static final Map<Tuple3<IterationID, Integer, String>, Object> m =
            new ConcurrentHashMap<>();

    private static final Map<Tuple3<IterationID, Integer, String>, OperatorID> owners =
            new ConcurrentHashMap<>();

    /**
     * Gets a {@link Reader} of shared data identified by (iterationId, subtaskId, key).
     *
     * @param iterationID The iteration ID.
     * @param subtaskId The subtask ID.
     * @param key The string key.
     * @return A {@link Reader} of shared data.
     * @param <T> The type of shared ata.
     */
    public static <T> Reader<T> getReader(IterationID iterationID, int subtaskId, String key) {
        return new Reader<>(Tuple3.of(iterationID, subtaskId, key));
    }

    /**
     * Gets a {@link Writer} of shared data identified by (iterationId, subtaskId, key).
     *
     * @param iterationID The iteration ID.
     * @param subtaskId The subtask ID.
     * @param key The string key.
     * @param operatorID The owner operator.
     * @param initVal Initialize value of the data.
     * @return A {@link Writer} of shared data.
     * @param <T> The type of shared ata.
     */
    public static <T> Writer<T> getWriter(
            IterationID iterationID, int subtaskId, String key, OperatorID operatorID, T initVal) {
        Tuple3<IterationID, Integer, String> t = Tuple3.of(iterationID, subtaskId, key);
        OperatorID lastOwner = owners.putIfAbsent(t, operatorID);
        if (null != lastOwner) {
            throw new IllegalStateException(
                    String.format(
                            "The shared data (%s, %s, %s) already has a writer %s.",
                            iterationID, subtaskId, key, operatorID));
        }
        Writer<T> writer = new Writer<>(t, operatorID);
        writer.set(initVal);
        return writer;
    }

    /**
     * A reader of shared data identified by key (IterationID, subtaskID, key).
     *
     * @param <T> The type of shared ata.
     */
    public static class Reader<T> {
        protected final Tuple3<IterationID, Integer, String> t;

        public Reader(Tuple3<IterationID, Integer, String> t) {
            this.t = t;
        }

        /**
         * Get the value.
         *
         * @return The value.
         */
        public T get() {
            //noinspection unchecked
            return (T) m.get(t);
        }
    }

    /**
     * A writer of shared data identified by key (IterationID, subtaskID, key). A writer is
     * responsible for the checkpointing of data.
     *
     * @param <T> The type of shared ata.
     */
    public static class Writer<T> extends Reader<T> {
        private final OperatorID operatorID;

        public Writer(Tuple3<IterationID, Integer, String> t, OperatorID operatorID) {
            super(t);
            this.operatorID = operatorID;
        }

        private void ensureOwner() {
            // Double-checks the owner, because a writer may call this method after the key removed
            // and re-added by other operators.
            Preconditions.checkState(owners.get(t).equals(operatorID));
        }

        /**
         * Set new value.
         *
         * @param value The new value.
         */
        public void set(T value) {
            ensureOwner();
            m.put(t, value);
        }

        /** Remove this data entry. */
        public void remove() {
            ensureOwner();
            m.remove(t);
            owners.remove(t);
        }
    }
}
