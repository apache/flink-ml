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

package org.apache.flink.iteration.proxy.state;

import org.apache.flink.runtime.state.CheckpointableKeyedStateBackend;
import org.apache.flink.runtime.state.KeyGroupStatePartitionStreamProvider;
import org.apache.flink.runtime.state.OperatorStateBackend;
import org.apache.flink.runtime.state.StatePartitionStreamProvider;
import org.apache.flink.streaming.api.operators.InternalTimeServiceManager;
import org.apache.flink.streaming.api.operators.StreamOperatorStateContext;
import org.apache.flink.util.CloseableIterable;

import java.util.Objects;
import java.util.OptionalLong;

/** Proxy {@link StreamOperatorStateContext} for the wrapped operator. */
public class ProxyStreamOperatorStateContext implements StreamOperatorStateContext {

    private final StreamOperatorStateContext wrapped;

    private final String stateNamePrefix;

    public ProxyStreamOperatorStateContext(
            StreamOperatorStateContext wrapped, String stateNamePrefix) {
        this.wrapped = Objects.requireNonNull(wrapped);
        this.stateNamePrefix = stateNamePrefix;
    }

    @Override
    public boolean isRestored() {
        return wrapped.isRestored();
    }

    @Override
    public OptionalLong getRestoredCheckpointId() {
        return wrapped.getRestoredCheckpointId();
    }

    @Override
    public OperatorStateBackend operatorStateBackend() {
        return wrapped.operatorStateBackend() == null
                ? null
                : new ProxyOperatorStateBackend(wrapped.operatorStateBackend(), stateNamePrefix);
    }

    @Override
    public CheckpointableKeyedStateBackend<?> keyedStateBackend() {
        return wrapped.keyedStateBackend() == null
                ? null
                : new ProxyKeyedStateBackend<>(wrapped.keyedStateBackend(), stateNamePrefix);
    }

    @Override
    public InternalTimeServiceManager<?> internalTimerServiceManager() {
        return wrapped.internalTimerServiceManager() == null
                ? null
                : new ProxyInternalTimeServiceManager<>(
                        wrapped.internalTimerServiceManager(), stateNamePrefix);
    }

    @Override
    public CloseableIterable<StatePartitionStreamProvider> rawOperatorStateInputs() {
        return CloseableIterable.empty();
    }

    @Override
    public CloseableIterable<KeyGroupStatePartitionStreamProvider> rawKeyedStateInputs() {
        return CloseableIterable.empty();
    }
}
