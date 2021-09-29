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

package org.apache.flink.ml.iteration.proxy.state;

import org.apache.flink.api.common.state.AggregatingStateDescriptor;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ReducingStateDescriptor;
import org.apache.flink.api.common.state.State;
import org.apache.flink.api.common.state.StateDescriptor;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.state.CheckpointStreamFactory;
import org.apache.flink.runtime.state.CheckpointableKeyedStateBackend;
import org.apache.flink.runtime.state.KeyGroupRange;
import org.apache.flink.runtime.state.KeyGroupedInternalPriorityQueue;
import org.apache.flink.runtime.state.Keyed;
import org.apache.flink.runtime.state.KeyedStateBackend;
import org.apache.flink.runtime.state.KeyedStateFunction;
import org.apache.flink.runtime.state.KeyedStateHandle;
import org.apache.flink.runtime.state.PriorityComparable;
import org.apache.flink.runtime.state.SavepointResources;
import org.apache.flink.runtime.state.SnapshotResult;
import org.apache.flink.runtime.state.StateSnapshotTransformer;
import org.apache.flink.runtime.state.heap.HeapPriorityQueueElement;

import javax.annotation.Nonnull;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.RunnableFuture;
import java.util.stream.Stream;

/** Proxy {@link KeyedStateBackend} for the wrapped operators. */
public class ProxyKeyedStateBackend<K> implements CheckpointableKeyedStateBackend<K> {

    private final CheckpointableKeyedStateBackend<K> wrappedBackend;

    private final String stateNamePrefix;

    public ProxyKeyedStateBackend(
            CheckpointableKeyedStateBackend<K> wrappedBackend, String stateNamePrefix) {
        this.wrappedBackend = wrappedBackend;
        this.stateNamePrefix = stateNamePrefix;
    }

    @Override
    public void setCurrentKey(K newKey) {
        wrappedBackend.setCurrentKey(newKey);
    }

    @Override
    public K getCurrentKey() {
        return wrappedBackend.getCurrentKey();
    }

    @Override
    public TypeSerializer<K> getKeySerializer() {
        return wrappedBackend.getKeySerializer();
    }

    @Override
    public <N, S extends State, T> void applyToAllKeys(
            N namespace,
            TypeSerializer<N> namespaceSerializer,
            StateDescriptor<S, T> stateDescriptor,
            KeyedStateFunction<K, S> function)
            throws Exception {
        StateDescriptor<S, T> newDescriptor = createNewDescriptor(stateDescriptor);
        wrappedBackend.applyToAllKeys(namespace, namespaceSerializer, newDescriptor, function);
    }

    @Override
    public <N> Stream<K> getKeys(String state, N namespace) {
        return wrappedBackend.getKeys(stateNamePrefix + state, namespace);
    }

    @Override
    public <N> Stream<Tuple2<K, N>> getKeysAndNamespaces(String state) {
        return wrappedBackend.getKeysAndNamespaces(stateNamePrefix + state);
    }

    @Override
    public <N, S extends State, T> S getOrCreateKeyedState(
            TypeSerializer<N> namespaceSerializer, StateDescriptor<S, T> stateDescriptor)
            throws Exception {
        StateDescriptor<S, T> newDescriptor = createNewDescriptor(stateDescriptor);
        return wrappedBackend.getOrCreateKeyedState(namespaceSerializer, newDescriptor);
    }

    @Override
    public <N, S extends State> S getPartitionedState(
            N namespace,
            TypeSerializer<N> namespaceSerializer,
            StateDescriptor<S, ?> stateDescriptor)
            throws Exception {
        StateDescriptor<S, ?> newDescriptor = createNewDescriptor(stateDescriptor);
        return wrappedBackend.getPartitionedState(namespace, namespaceSerializer, newDescriptor);
    }

    @Override
    public void registerKeySelectionListener(KeySelectionListener<K> listener) {
        wrappedBackend.registerKeySelectionListener(listener);
    }

    @Override
    public boolean deregisterKeySelectionListener(KeySelectionListener<K> listener) {
        return wrappedBackend.deregisterKeySelectionListener(listener);
    }

    @Nonnull
    @Override
    public <N, SV, SEV, S extends State, IS extends S> IS createInternalState(
            @Nonnull TypeSerializer<N> namespaceSerializer,
            @Nonnull StateDescriptor<S, SV> stateDesc,
            @Nonnull
                    StateSnapshotTransformer.StateSnapshotTransformFactory<SEV>
                            snapshotTransformFactory)
            throws Exception {
        StateDescriptor<S, ?> newDescriptor = createNewDescriptor(stateDesc);
        return wrappedBackend.createInternalState(
                namespaceSerializer, newDescriptor, snapshotTransformFactory);
    }

    @SuppressWarnings("unchecked")
    protected <S extends State, T> StateDescriptor<S, T> createNewDescriptor(
            StateDescriptor<S, T> descriptor) {
        switch (descriptor.getType()) {
            case VALUE:
                {
                    return (StateDescriptor<S, T>)
                            new ValueStateDescriptor<>(
                                    stateNamePrefix + descriptor.getName(),
                                    descriptor.getSerializer());
                }
            case LIST:
                {
                    ListStateDescriptor<T> listStateDescriptor =
                            (ListStateDescriptor<T>) descriptor;
                    return (StateDescriptor<S, T>)
                            new ListStateDescriptor<>(
                                    stateNamePrefix + listStateDescriptor.getName(),
                                    listStateDescriptor.getElementSerializer());
                }
            case REDUCING:
                {
                    ReducingStateDescriptor<T> reducingStateDescriptor =
                            (ReducingStateDescriptor<T>) descriptor;
                    return (StateDescriptor<S, T>)
                            new ReducingStateDescriptor<>(
                                    stateNamePrefix + reducingStateDescriptor.getName(),
                                    reducingStateDescriptor.getReduceFunction(),
                                    reducingStateDescriptor.getSerializer());
                }
            case AGGREGATING:
                {
                    AggregatingStateDescriptor<?, ?, T> aggregatingStateDescriptor =
                            (AggregatingStateDescriptor<?, ?, T>) descriptor;
                    return new AggregatingStateDescriptor(
                            stateNamePrefix + aggregatingStateDescriptor.getName(),
                            aggregatingStateDescriptor.getAggregateFunction(),
                            aggregatingStateDescriptor.getSerializer());
                }
            case MAP:
                {
                    MapStateDescriptor<?, Map<?, ?>> mapStateDescriptor =
                            (MapStateDescriptor<?, Map<?, ?>>) descriptor;
                    return new MapStateDescriptor(
                            stateNamePrefix + mapStateDescriptor.getName(),
                            mapStateDescriptor.getKeySerializer(),
                            mapStateDescriptor.getValueSerializer());
                }
            default:
                throw new UnsupportedOperationException("Unsupported state type");
        }
    }

    @Override
    public KeyGroupRange getKeyGroupRange() {
        return wrappedBackend.getKeyGroupRange();
    }

    @Nonnull
    @Override
    public SavepointResources<K> savepoint() throws Exception {
        return wrappedBackend.savepoint();
    }

    @Override
    public void dispose() {
        // Do not dispose for poxy.
    }

    @Override
    public void close() throws IOException {
        // Do not close for poxy.
    }

    @Nonnull
    @Override
    public <T extends HeapPriorityQueueElement & PriorityComparable<? super T> & Keyed<?>>
            KeyGroupedInternalPriorityQueue<T> create(
                    @Nonnull String stateName,
                    @Nonnull TypeSerializer<T> byteOrderedElementSerializer) {
        return wrappedBackend.create(stateNamePrefix + stateName, byteOrderedElementSerializer);
    }

    @Nonnull
    @Override
    public RunnableFuture<SnapshotResult<KeyedStateHandle>> snapshot(
            long checkpointId,
            long timestamp,
            @Nonnull CheckpointStreamFactory streamFactory,
            @Nonnull CheckpointOptions checkpointOptions)
            throws Exception {
        return wrappedBackend.snapshot(checkpointId, timestamp, streamFactory, checkpointOptions);
    }
}
