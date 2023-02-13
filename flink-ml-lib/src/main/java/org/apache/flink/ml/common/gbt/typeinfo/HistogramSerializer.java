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

package org.apache.flink.ml.common.gbt.typeinfo;

import org.apache.flink.api.common.typeutils.SimpleTypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.api.common.typeutils.base.array.DoublePrimitiveArraySerializer;
import org.apache.flink.api.common.typeutils.base.array.IntPrimitiveArraySerializer;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.Histogram;

import java.io.IOException;

/** Serializer for {@link Histogram}. */
public final class HistogramSerializer extends TypeSerializerSingleton<Histogram> {

    public static final HistogramSerializer INSTANCE = new HistogramSerializer();
    private static final long serialVersionUID = 1L;

    private static final SplitSerializer SPLIT_SERIALIZER = SplitSerializer.INSTANCE;

    @Override
    public boolean isImmutableType() {
        return false;
    }

    @Override
    public Histogram createInstance() {
        return new Histogram();
    }

    @Override
    public Histogram copy(Histogram from) {
        Histogram histogram = new Histogram();
        histogram.subtaskId = from.subtaskId;
        histogram.hists = from.hists.clone();
        histogram.recvcnts = from.recvcnts.clone();
        return histogram;
    }

    @Override
    public Histogram copy(Histogram from, Histogram reuse) {
        assert from.getClass() == reuse.getClass();
        reuse.subtaskId = from.subtaskId;
        reuse.hists = from.hists.clone();
        reuse.recvcnts = from.recvcnts.clone();
        return reuse;
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Histogram record, DataOutputView target) throws IOException {
        IntSerializer.INSTANCE.serialize(record.subtaskId, target);
        DoublePrimitiveArraySerializer.INSTANCE.serialize(record.hists, target);
        IntPrimitiveArraySerializer.INSTANCE.serialize(record.recvcnts, target);
    }

    @Override
    public Histogram deserialize(DataInputView source) throws IOException {
        Histogram histogram = new Histogram();
        histogram.subtaskId = IntSerializer.INSTANCE.deserialize(source);
        histogram.hists = DoublePrimitiveArraySerializer.INSTANCE.deserialize(source);
        histogram.recvcnts = IntPrimitiveArraySerializer.INSTANCE.deserialize(source);
        return histogram;
    }

    @Override
    public Histogram deserialize(Histogram reuse, DataInputView source) throws IOException {
        reuse.subtaskId = IntSerializer.INSTANCE.deserialize(source);
        reuse.hists = DoublePrimitiveArraySerializer.INSTANCE.deserialize(source);
        reuse.recvcnts = IntPrimitiveArraySerializer.INSTANCE.deserialize(source);
        return reuse;
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        serialize(deserialize(source), target);
    }

    // ------------------------------------------------------------------------

    @Override
    public TypeSerializerSnapshot<Histogram> snapshotConfiguration() {
        return new HistogramSerializerSnapshot();
    }

    /** Serializer configuration snapshot for compatibility and format evolution. */
    @SuppressWarnings("WeakerAccess")
    public static final class HistogramSerializerSnapshot
            extends SimpleTypeSerializerSnapshot<Histogram> {

        public HistogramSerializerSnapshot() {
            super(HistogramSerializer::new);
        }
    }
}
