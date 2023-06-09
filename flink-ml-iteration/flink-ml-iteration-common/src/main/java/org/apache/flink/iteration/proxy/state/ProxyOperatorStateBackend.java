/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.iteration.proxy.state;

import org.apache.flink.api.common.state.BroadcastState;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.StateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.ListTypeInfo;
import org.apache.flink.api.java.typeutils.MapTypeInfo;
import org.apache.flink.iteration.utils.ReflectionUtils;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.runtime.state.OperatorStateBackend;
import org.apache.flink.runtime.state.OperatorStateHandle;
import org.apache.flink.runtime.state.SnapshotResult;

import javax.annotation.Nonnull;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.RunnableFuture;

/** Proxy {@link OperatorStateBackend} for the wrapped Operator. */
public class ProxyOperatorStateBackend implements OperatorStateBackend {

    private final OperatorStateBackend wrappedBackend;

    private final StateNamePrefix stateNamePrefix;

    public ProxyOperatorStateBackend(
            OperatorStateBackend wrappedBackend, StateNamePrefix stateNamePrefix) {
        this.wrappedBackend = wrappedBackend;
        this.stateNamePrefix = stateNamePrefix;
    }

    @Override
    public <K, V> BroadcastState<K, V> getBroadcastState(MapStateDescriptor<K, V> stateDescriptor)
            throws Exception {
        MapStateDescriptor<K, V> newDescriptor;
        if (stateDescriptor.isSerializerInitialized()) {
            newDescriptor =
                    new MapStateDescriptor<>(
                            stateNamePrefix.prefix(stateDescriptor.getName()),
                            stateDescriptor.getKeySerializer(),
                            stateDescriptor.getValueSerializer());
        } else {
            MapTypeInfo<K, V> mapTypeInfo = getMapTypeInfo(stateDescriptor);
            newDescriptor =
                    new MapStateDescriptor<>(
                            stateNamePrefix.prefix(stateDescriptor.getName()),
                            mapTypeInfo.getKeyTypeInfo(),
                            mapTypeInfo.getValueTypeInfo());
        }
        return wrappedBackend.getBroadcastState(newDescriptor);
    }

    @Override
    public <S> ListState<S> getListState(ListStateDescriptor<S> stateDescriptor) throws Exception {
        ListStateDescriptor<S> newDescriptor =
                stateDescriptor.isSerializerInitialized()
                        ? new ListStateDescriptor<>(
                                stateNamePrefix.prefix(stateDescriptor.getName()),
                                stateDescriptor.getElementSerializer())
                        : new ListStateDescriptor<>(
                                stateNamePrefix.prefix(stateDescriptor.getName()),
                                getElementTypeInfo(stateDescriptor));

        return wrappedBackend.getListState(newDescriptor);
    }

    @Override
    public <S> ListState<S> getUnionListState(ListStateDescriptor<S> stateDescriptor)
            throws Exception {
        ListStateDescriptor<S> newDescriptor =
                stateDescriptor.isSerializerInitialized()
                        ? new ListStateDescriptor<>(
                                stateNamePrefix.prefix(stateDescriptor.getName()),
                                stateDescriptor.getElementSerializer())
                        : new ListStateDescriptor<>(
                                stateNamePrefix.prefix(stateDescriptor.getName()),
                                getElementTypeInfo(stateDescriptor));
        return wrappedBackend.getUnionListState(newDescriptor);
    }

    @Override
    public Set<String> getRegisteredStateNames() {
        Set<String> filteredNames = new HashSet<>();
        Set<String> names = wrappedBackend.getRegisteredStateNames();

        for (String name : names) {
            if (name.startsWith(stateNamePrefix.getNamePrefix())) {
                filteredNames.add(name.substring(stateNamePrefix.getNamePrefix().length()));
            }
        }

        return filteredNames;
    }

    @Override
    public Set<String> getRegisteredBroadcastStateNames() {
        Set<String> filteredNames = new HashSet<>();
        Set<String> names = wrappedBackend.getRegisteredBroadcastStateNames();

        for (String name : names) {
            if (name.startsWith(stateNamePrefix.getNamePrefix())) {
                filteredNames.add(name.substring(stateNamePrefix.getNamePrefix().length()));
            }
        }

        return filteredNames;
    }

    @Override
    public void dispose() {
        // Do not dispose for proxy.
    }

    @Override
    public void close() throws IOException {
        // Do not close for proxy.
    }

    @Nonnull
    @Override
    public RunnableFuture<SnapshotResult<OperatorStateHandle>> snapshot(
            long checkpointId,
            long timestamp,
            @Nonnull CheckpointStreamFactory streamFactory,
            @Nonnull CheckpointOptions checkpointOptions)
            throws Exception {
        return wrappedBackend.snapshot(checkpointId, timestamp, streamFactory, checkpointOptions);
    }

    @SuppressWarnings("unchecked,rawtypes")
    private <S> TypeInformation<S> getElementTypeInfo(ListStateDescriptor<S> stateDescriptor) {
        return ((ListTypeInfo)
                        ReflectionUtils.getFieldValue(
                                stateDescriptor, StateDescriptor.class, "typeInfo"))
                .getElementTypeInfo();
    }

    private <K, V> MapTypeInfo<K, V> getMapTypeInfo(MapStateDescriptor<K, V> stateDescriptor) {
        return ReflectionUtils.getFieldValue(stateDescriptor, StateDescriptor.class, "typeInfo");
    }
}
