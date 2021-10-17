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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.common.operators.MailboxExecutor;
import org.apache.flink.api.java.tuple.Tuple2;

import javax.annotation.Nullable;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Context to hold the broadcast variables and provides some utility function for accessing
 * broadcast variables.
 */
public class BroadcastContext {

    /**
     * stores broadcast data streams in a map. The key is broadcastName-partitionId and the value is
     * {@link BroadcastItem}.
     */
    private static final ConcurrentHashMap<String, BroadcastItem> BROADCAST_VARIABLES =
            new ConcurrentHashMap<>();

    @VisibleForTesting
    public static void putBroadcastVariable(String key, Tuple2<Boolean, List<?>> variable) {
        BROADCAST_VARIABLES.compute(
                key,
                (k, v) ->
                        null == v
                                ? new BroadcastItem(variable.f0, variable.f1, null)
                                : new BroadcastItem(variable.f0, variable.f1, v.mailboxExecutor));
    }

    @VisibleForTesting
    public static void putMailBoxExecutor(String key, MailboxExecutor mailboxExecutor) {
        BROADCAST_VARIABLES.compute(
                key,
                (k, v) ->
                        null == v
                                ? new BroadcastItem(false, null, mailboxExecutor)
                                : new BroadcastItem(v.cacheReady, v.cacheList, mailboxExecutor));
    }

    @VisibleForTesting
    @SuppressWarnings({"unchecked"})
    public static <T> List<T> getBroadcastVariable(String key) {
        return (List<T>) BROADCAST_VARIABLES.get(key).cacheList;
    }

    @VisibleForTesting
    public static void remove(String key) {
        BROADCAST_VARIABLES.remove(key);
    }

    @VisibleForTesting
    public static void markCacheFinished(String key) {
        BROADCAST_VARIABLES.computeIfPresent(
                key,
                (k, v) -> {
                    // sends an dummy email to avoid possible stuck.
                    if (null != v.mailboxExecutor) {
                        v.mailboxExecutor.execute(() -> {}, "empty mail");
                    }
                    return new BroadcastItem(true, v.cacheList, v.mailboxExecutor);
                });
    }

    @VisibleForTesting
    public static boolean isCacheFinished(String key) {
        return BROADCAST_VARIABLES.get(key).cacheReady;
    }

    /** Utility class to organize broadcast variables. */
    private static class BroadcastItem {

        /** whether this broadcast variable is ready to be consumed. */
        private boolean cacheReady;

        /** the cached list */
        private List<?> cacheList;

        /** the mailboxExecutor of the consumer, used to avoid the possible stuck of consumer. */
        private MailboxExecutor mailboxExecutor;

        BroadcastItem(
                boolean cacheReady,
                @Nullable List<?> cacheList,
                @Nullable MailboxExecutor mailboxExecutor) {
            this.cacheReady = cacheReady;
            this.cacheList = cacheList;
            this.mailboxExecutor = mailboxExecutor;
        }
    }
}
