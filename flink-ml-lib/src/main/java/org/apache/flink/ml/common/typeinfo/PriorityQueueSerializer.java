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

package org.apache.flink.ml.common.typeinfo;

import org.apache.flink.api.common.typeutils.CompositeTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.ListSerializer;
import org.apache.flink.api.java.typeutils.runtime.DataInputViewStream;
import org.apache.flink.api.java.typeutils.runtime.DataOutputViewStream;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.util.InstantiationUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * TypeSerializer for {@link java.util.PriorityQueue}.
 *
 * @param <T> The type of elements in the PriorityQueue.
 */
public class PriorityQueueSerializer<T> extends TypeSerializer<PriorityQueue<T>> {

    private final Comparator<? super T> comparator;

    private final TypeSerializer<T> elementSerializer;

    public PriorityQueueSerializer(
            Comparator<? super T> comparator, TypeSerializer<T> elementSerializer) {
        this.comparator = comparator;
        this.elementSerializer = elementSerializer;
    }

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public TypeSerializer<PriorityQueue<T>> duplicate() {
        return new PriorityQueueSerializer<>(comparator, elementSerializer.duplicate());
    }

    @Override
    public PriorityQueue<T> createInstance() {
        return new PriorityQueue<>(comparator);
    }

    @Override
    public PriorityQueue<T> copy(PriorityQueue<T> from) {
        return new PriorityQueue<>(from);
    }

    @Override
    public PriorityQueue<T> copy(PriorityQueue<T> from, PriorityQueue<T> reuse) {
        return new PriorityQueue<>(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(PriorityQueue<T> queue, DataOutputView target) throws IOException {
        List<T> tmpList = new ArrayList<>(queue);
        ListSerializer<T> listSerializer = new ListSerializer<>(elementSerializer);
        listSerializer.serialize(tmpList, target);
    }

    @Override
    public PriorityQueue<T> deserialize(DataInputView source) throws IOException {
        ListSerializer<T> listSerializer = new ListSerializer<>(elementSerializer);
        List<T> tmpList = listSerializer.deserialize(source);
        PriorityQueue<T> queue = new PriorityQueue<>(comparator);
        queue.addAll(tmpList);
        return queue;
    }

    @Override
    public PriorityQueue<T> deserialize(PriorityQueue<T> reuse, DataInputView source)
            throws IOException {
        return deserialize(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        ListSerializer<T> listSerializer = new ListSerializer<>(elementSerializer);
        listSerializer.copy(source, target);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        PriorityQueueSerializer<?> that = (PriorityQueueSerializer<?>) o;
        return Objects.equals(elementSerializer, that.elementSerializer)
                && Objects.equals(comparator, that.comparator);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                comparator != null ? comparator.hashCode() : 0,
                elementSerializer != null ? elementSerializer.hashCode() : 0);
    }

    @Override
    public TypeSerializerSnapshot<PriorityQueue<T>> snapshotConfiguration() {
        return new PriorityQueueTypeSerializerSnapshot<>(this);
    }

    private static final class PriorityQueueTypeSerializerSnapshot<T>
            extends CompositeTypeSerializerSnapshot<PriorityQueue<T>, PriorityQueueSerializer<T>> {

        private static final int CURRENT_VERSION = 1;

        private Comparator<? super T> comparator;

        public PriorityQueueTypeSerializerSnapshot() {
            super(PriorityQueueSerializer.class);
        }

        public PriorityQueueTypeSerializerSnapshot(PriorityQueueSerializer<T> serializer) {
            super(serializer);
            this.comparator = serializer.comparator;
        }

        @Override
        protected int getCurrentOuterSnapshotVersion() {
            return CURRENT_VERSION;
        }

        @Override
        protected TypeSerializer<?>[] getNestedSerializers(
                PriorityQueueSerializer<T> outerSerializer) {
            return new TypeSerializer[] {outerSerializer.elementSerializer};
        }

        @Override
        protected PriorityQueueSerializer<T> createOuterSerializerWithNestedSerializers(
                TypeSerializer<?>[] nestedSerializers) {
            TypeSerializer<T> elementSerializer = (TypeSerializer<T>) nestedSerializers[0];
            return new PriorityQueueSerializer<>(comparator, elementSerializer);
        }

        @Override
        protected void writeOuterSnapshot(DataOutputView out) throws IOException {
            final DataOutputViewStream stream = new DataOutputViewStream(out);
            InstantiationUtil.serializeObject(stream, comparator);
        }

        @Override
        protected void readOuterSnapshot(
                int readOuterSnapshotVersion, DataInputView in, ClassLoader userCodeClassLoader)
                throws IOException {
            final DataInputViewStream stream = new DataInputViewStream(in);
            try {
                comparator = InstantiationUtil.deserializeObject(stream, userCodeClassLoader);
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }

        @Override
        protected OuterSchemaCompatibility resolveOuterSchemaCompatibility(
                PriorityQueueSerializer<T> newSerializer) {
            return (this.comparator.getClass() == newSerializer.comparator.getClass())
                    ? OuterSchemaCompatibility.COMPATIBLE_AS_IS
                    : OuterSchemaCompatibility.INCOMPATIBLE;
        }
    }
}
