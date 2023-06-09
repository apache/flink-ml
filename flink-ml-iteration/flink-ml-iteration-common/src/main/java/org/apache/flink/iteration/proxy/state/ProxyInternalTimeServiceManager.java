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

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.runtime.state.KeyedStateCheckpointOutputStream;
import org.apache.flink.streaming.api.operators.InternalTimeServiceManager;
import org.apache.flink.streaming.api.operators.InternalTimerService;
import org.apache.flink.streaming.api.operators.Triggerable;
import org.apache.flink.streaming.api.watermark.Watermark;

/** Proxy {@link InternalTimeServiceManager} for the wrapped operators. */
public class ProxyInternalTimeServiceManager<K> implements InternalTimeServiceManager<K> {

    private final InternalTimeServiceManager<K> wrappedManager;

    private final StateNamePrefix stateNamePrefix;

    public ProxyInternalTimeServiceManager(
            InternalTimeServiceManager<K> wrappedManager, StateNamePrefix stateNamePrefix) {
        this.wrappedManager = wrappedManager;
        this.stateNamePrefix = stateNamePrefix;
    }

    @Override
    public <N> InternalTimerService<N> getInternalTimerService(
            String name,
            TypeSerializer<K> keySerializer,
            TypeSerializer<N> namespaceSerializer,
            Triggerable<K, N> triggerable) {
        return wrappedManager.getInternalTimerService(
                stateNamePrefix.prefix(name), keySerializer, namespaceSerializer, triggerable);
    }

    @Override
    public void advanceWatermark(Watermark watermark) throws Exception {
        wrappedManager.advanceWatermark(watermark);
    }

    @Override
    public void snapshotToRawKeyedState(
            KeyedStateCheckpointOutputStream stateCheckpointOutputStream, String operatorName)
            throws Exception {
        wrappedManager.snapshotToRawKeyedState(stateCheckpointOutputStream, operatorName);
    }
}
