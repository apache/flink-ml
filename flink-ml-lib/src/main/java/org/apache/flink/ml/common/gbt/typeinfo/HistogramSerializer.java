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
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.ml.common.gbt.defs.Histogram;
import org.apache.flink.ml.common.gbt.defs.Slice;
import org.apache.flink.ml.linalg.typeinfo.OptimizedDoublePrimitiveArraySerializer;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;

/** Serializer for {@link Histogram}. */
public final class HistogramSerializer extends TypeSerializerSingleton<Histogram> {

    public static final HistogramSerializer INSTANCE = new HistogramSerializer();
    private static final long serialVersionUID = 1L;

    private final OptimizedDoublePrimitiveArraySerializer histsSerializer =
            new OptimizedDoublePrimitiveArraySerializer();

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
        histogram.hists = ArrayUtils.subarray(from.hists, from.slice.start, from.slice.end);
        histogram.slice.start = 0;
        histogram.slice.end = from.slice.size();
        return histogram;
    }

    @Override
    public Histogram copy(Histogram from, Histogram reuse) {
        return copy(from);
    }

    @Override
    public int getLength() {
        return -1;
    }

    @Override
    public void serialize(Histogram record, DataOutputView target) throws IOException {
        // Only writes valid slice of `hists`.
        histsSerializer.serialize(record.hists, record.slice.start, record.slice.size(), target);
    }

    @Override
    public Histogram deserialize(DataInputView source) throws IOException {
        Histogram histogram = new Histogram();
        histogram.hists = histsSerializer.deserialize(source);
        histogram.slice = new Slice(0, histogram.hists.length);
        return histogram;
    }

    @Override
    public Histogram deserialize(Histogram reuse, DataInputView source) throws IOException {
        reuse.hists = histsSerializer.deserialize(reuse.hists, source);
        reuse.slice.start = 0;
        reuse.slice.end = reuse.hists.length;
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
