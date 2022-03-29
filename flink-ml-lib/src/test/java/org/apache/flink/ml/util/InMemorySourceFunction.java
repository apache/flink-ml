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

package org.apache.flink.ml.util;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.source.RichSourceFunction;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.util.Preconditions;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

/** A {@link SourceFunction} implementation that can directly receive records from tests. */
@SuppressWarnings({"unchecked", "rawtypes"})
public class InMemorySourceFunction<T> extends RichSourceFunction<T> {
    private static final Map<UUID, BlockingQueue> queueMap = new ConcurrentHashMap<>();
    private final UUID id;
    private BlockingQueue<Optional<T>> queue;
    private volatile boolean isRunning = true;

    public InMemorySourceFunction() {
        id = UUID.randomUUID();
        queue = new LinkedBlockingQueue();
        queueMap.put(id, queue);
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        queue = queueMap.get(id);
    }

    @Override
    public void close() throws Exception {
        super.close();
        queueMap.remove(id);
    }

    @Override
    public void run(SourceContext<T> context) throws InterruptedException {
        while (isRunning) {
            Optional<T> maybeValue = queue.take();
            if (!maybeValue.isPresent()) {
                break;
            }
            context.collect(maybeValue.get());
        }
    }

    @Override
    public void cancel() {
        isRunning = false;
        queue.add(Optional.empty());
    }

    @SafeVarargs
    public final void addAll(T... values) {
        Preconditions.checkState(isRunning);
        for (T value : values) {
            queue.add(Optional.of(value));
        }
    }
}
