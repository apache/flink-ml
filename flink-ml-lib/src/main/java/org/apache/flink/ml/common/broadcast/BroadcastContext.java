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

package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.java.tuple.Tuple2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class BroadcastContext {
    /**
     * Store broadcast DataStreams in a Map. The key is (broadcastName, partitionId) and the value
     * is (isBroaddcastVariableReady, cacheList).
     */
    private static Map<Tuple2<String, Integer>, Tuple2<Boolean, List<?>>> broadcastVariables =
            new HashMap<>();
    /**
     * We use lock because we want to enable `getBroadcastVariable(String)` in a TM with multiple
     * slots here. Note that using ConcurrentHashMap is not enough since we need "contains and get
     * in an atomic operation".
     */
    private static ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    public static void putBroadcastVariable(
            Tuple2<String, Integer> key, Tuple2<Boolean, List<?>> variable) {
        lock.writeLock().lock();
        try {
            broadcastVariables.put(key, variable);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * get the cached list with the given key.
     *
     * @param key
     * @param <T>
     * @return the cache list.
     */
    public static <T> List<T> getBroadcastVariable(Tuple2<String, Integer> key) {
        lock.readLock().lock();
        List<?> result = null;
        try {
            result = broadcastVariables.get(key).f1;
        } finally {
            lock.readLock().unlock();
        }
        return (List<T>) result;
    }

    /**
     * get broadcast variables by name
     *
     * @param name
     * @param <T>
     * @return
     */
    public static <T> List<T> getBroadcastVariable(String name) {
        lock.readLock().lock();
        List<?> result = null;
        try {
            for (Tuple2<String, Integer> nameAndPartitionId : broadcastVariables.keySet()) {
                if (name.equals(nameAndPartitionId.f0) && isCacheFinished(nameAndPartitionId)) {
                    result = broadcastVariables.get(nameAndPartitionId).f1;
                    break;
                }
            }
        } finally {
            lock.readLock().unlock();
        }
        return (List<T>) result;
    }

    public static void remove(Tuple2<String, Integer> key) {
        lock.writeLock().lock();
        try {
            broadcastVariables.remove(key);
        } finally {
            lock.writeLock().unlock();
        }
    }

    public static void markCacheFinished(Tuple2<String, Integer> key) {
        lock.writeLock().lock();
        try {
            broadcastVariables.get(key).f0 = true;
        } finally {
            lock.writeLock().unlock();
        }
    }

    public static boolean isCacheFinished(Tuple2<String, Integer> key) {
        lock.readLock().lock();
        boolean isFinished = false;
        try {
            if (broadcastVariables.containsKey(key)) {
                isFinished = broadcastVariables.get(key).f0;
            }
        } finally {
            lock.readLock().unlock();
        }
        return isFinished;
    }
}
