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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.PrimitiveArrayTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.common.ps.typeinfo.Long2ObjectOpenHashMapTypeInfo;
import org.apache.flink.ml.common.ps.updater.ModelUpdater;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/** Als model updater supports the API for updating model and getting model. */
public class AlsModelUpdater implements ModelUpdater<Tuple2<Long, double[]>> {
    private final int rank;
    private final Random random = new Random();
    Long2ObjectOpenHashMap<double[]> model;
    private ListState<Long2ObjectOpenHashMap<double[]>> modelDataState;

    public AlsModelUpdater(int rank) {
        this.rank = rank;
    }

    @Override
    public void update(long[] keys, double[] values) {
        int offset = rank * (rank + 1);
        for (int i = 0; i < keys.length; ++i) {
            if (keys[i] >= 0) {
                model.put(keys[i], Arrays.copyOfRange(values, i * rank, (i + 1) * rank));
            } else {
                if (keys[i] == Long.MIN_VALUE || keys[i] == Long.MIN_VALUE + 1) {
                    continue;
                }
                assert (!model.containsKey(keys[i]));
                model.put(keys[i], Arrays.copyOfRange(values, i * offset, (i + 1) * offset));
            }
        }
    }

    @Override
    public double[] get(long[] keys) {
        if (keys[0] >= 0) {
            double[] values = new double[keys.length * rank];
            for (int i = 0; i < keys.length; i++) {
                if (!model.containsKey(keys[i])) {
                    double[] factor = new double[rank];
                    random.setSeed(keys[i]);
                    for (int j = 0; j < rank; ++j) {
                        factor[j] = random.nextDouble();
                    }
                    model.put(keys[i], factor);
                }
                System.arraycopy(model.get(keys[i]), 0, values, i * rank, rank);
            }
            return values;
        } else if (keys[0] == Long.MIN_VALUE) {
            return new double[rank];
        } else if (keys[0] == Long.MIN_VALUE + 1) {
            return new double[rank * rank + rank];
        } else {
            int offset = rank * (rank + 1);
            double[] values = new double[keys.length * offset];
            for (int i = 0; i < keys.length; i++) {
                if (keys[i] == Long.MIN_VALUE) {
                    continue;
                }
                System.arraycopy(model.get(keys[i]), 0, values, i * offset, offset);
                model.remove(keys[i]);
            }
            return values;
        }
    }

    @Override
    public Iterator<Tuple2<Long, double[]>> getModelSegments() {
        List<Tuple2<Long, double[]>> modelSegments = new ArrayList<>(model.size());
        for (Long key : model.keySet()) {
            if (key >= 0L) {
                modelSegments.add(Tuple2.of(key, model.get(key)));
            }
        }
        return modelSegments.iterator();
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        modelDataState =
                context.getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "modelDataState",
                                        new Long2ObjectOpenHashMapTypeInfo<>(
                                                PrimitiveArrayTypeInfo
                                                        .DOUBLE_PRIMITIVE_ARRAY_TYPE_INFO)));
        model =
                OperatorStateUtils.getUniqueElement(modelDataState, "modelDataState")
                        .orElse(new Long2ObjectOpenHashMap<>());
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        modelDataState.clear();
        modelDataState.add(model);
    }
}
