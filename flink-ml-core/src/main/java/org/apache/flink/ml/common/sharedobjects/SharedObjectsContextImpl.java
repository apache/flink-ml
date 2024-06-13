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

import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import static org.apache.flink.ml.common.sharedobjects.SharedObjectsPools.getReader;
import static org.apache.flink.ml.common.sharedobjects.SharedObjectsPools.getWriter;

/**
 * A default implementation of {@link SharedObjectsContext}.
 *
 * <p>It initializes readers and writers of shared objects according to the owner map when the
 * subtask starts and clean internal states when the subtask finishes. It also handles
 * `initializeState` and `snapshotState` automatically.
 */
@SuppressWarnings("rawtypes")
class SharedObjectsContextImpl implements SharedObjectsContext, Serializable {
    private final SharedObjectsPools.PoolID poolID;
    private final Map<Descriptor, SharedObjectsPools.Writer> writers = new HashMap<>();
    private final Map<Descriptor, SharedObjectsPools.Reader> readers = new HashMap<>();
    private Map<Descriptor<?>, String> ownerMap;

    /** The step of corresponding operator. See {@link ReadRequest} for more information. */
    private int step;

    public SharedObjectsContextImpl() {
        this.poolID = new SharedObjectsPools.PoolID();
        step = -1;
    }

    void setOwnerMap(Map<Descriptor<?>, String> ownerMap) {
        this.ownerMap = ownerMap;
    }

    void incStep(@Nullable Integer targetStep) {
        step += 1;
        // Sanity check
        Preconditions.checkState(null == targetStep || step == targetStep);
    }

    void incStep() {
        incStep(null);
    }

    void initializeState(
            StreamOperator<?> operator,
            StreamingRuntimeContext runtimeContext,
            StateInitializationContext context) {
        Preconditions.checkArgument(operator instanceof AbstractSharedObjectsStreamOperator);
        String ownerId = ((AbstractSharedObjectsStreamOperator) operator).getAccessorID();
        int subtaskId = runtimeContext.getIndexOfThisSubtask();
        for (Map.Entry<Descriptor<?>, String> entry : ownerMap.entrySet()) {
            Descriptor<?> descriptor = entry.getKey();
            if (ownerId.equals(entry.getValue())) {
                writers.put(
                        descriptor,
                        getWriter(
                                poolID,
                                subtaskId,
                                descriptor,
                                ownerId,
                                operator.getOperatorID(),
                                ((AbstractStreamOperator<?>) operator).getContainingTask(),
                                runtimeContext,
                                context,
                                step));
            }
            readers.put(descriptor, getReader(poolID, subtaskId, descriptor));
        }
    }

    void snapshotState(StateSnapshotContext context) throws Exception {
        for (SharedObjectsPools.Writer<?> writer : writers.values()) {
            writer.snapshotState(context);
        }
    }

    void clear() {
        for (SharedObjectsPools.Writer writer : writers.values()) {
            writer.remove();
        }
        for (SharedObjectsPools.Reader reader : readers.values()) {
            reader.remove();
        }
        writers.clear();
        readers.clear();
    }

    @Override
    public <T> T read(ReadRequest<T> request) {
        try {
            return read(request, false);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Gets the value of the shared object with possible waiting.
     *
     * @param request A read request of a shared object.
     * @param wait Whether to wait or not.
     * @return The value of the shared object, or null if not set yet.
     * @param <T> The type of the shared object.
     */
    <T> T read(ReadRequest<T> request, boolean wait) throws InterruptedException {
        Descriptor<T> descriptor = request.descriptor;
        //noinspection unchecked
        SharedObjectsPools.Reader<T> reader = readers.get(descriptor);
        switch (request.offset) {
            case SAME:
                return reader.get(step, wait);
            case PREV:
                return reader.get(step - 1, wait);
            case NEXT:
                return reader.get(step + 1, wait);
            default:
                throw new UnsupportedOperationException();
        }
    }

    @Override
    public <T> void write(Descriptor<T> descriptor, T value) {
        //noinspection unchecked
        SharedObjectsPools.Writer<T> writer = writers.get(descriptor);
        Preconditions.checkState(
                null != writer,
                String.format(
                        "The operator requestes to write a shared object %s not owned by itself.",
                        descriptor));
        writer.set(value, step);
    }

    @Override
    public <T> void renew(Descriptor<T> descriptor) {
        try {
            //noinspection unchecked
            write(
                    descriptor,
                    ((SharedObjectsPools.Reader<T>) readers.get(descriptor)).get(step - 1, false));
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
