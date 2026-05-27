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

package org.apache.flink.ml.common.ps.iterations;

import org.apache.flink.annotation.Internal;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.util.function.SerializableSupplier;

import java.io.Serializable;
import java.util.function.Supplier;

/**
 * An iteration stage that aggregates data from servers using keys as {@code PullStage#keys#get()}
 * and stores the aggregated values by {@code PullStage#values#get()}.
 *
 * <p>If the aggregator is null, we simply pull those values specified by the keys.
 */
public final class PullStage implements IterationStage {
    public final Supplier<SharedLongArray> keys;
    public final Supplier<SharedDoubleArray> values;
    public final Aggregator<double[], double[]> aggregator;

    public PullStage(
            SerializableSupplier<SharedLongArray> keys,
            SerializableSupplier<SharedDoubleArray> values) {
        this(keys, values, null);
    }

    public PullStage(
            SerializableSupplier<SharedLongArray> keys,
            SerializableSupplier<SharedDoubleArray> values,
            Aggregator<double[], double[]> aggregator) {
        this.keys = keys;
        this.values = values;
        this.aggregator = aggregator;
    }

    /**
     * An Aggregator is used to aggregate a set of input elements into a single accumulator.
     *
     * @param <IN> The type of the input elements.
     * @param <ACC> The type of the accumulator.
     */
    @Internal
    public interface Aggregator<IN, ACC> extends Serializable {

        /**
         * Adds a new input element to the given accumulator and returns the updated accumulator.
         *
         * @param in The input element to add.
         * @param acc The accumulator to update.
         * @return The updated accumulator.
         */
        ACC add(IN in, ACC acc);

        /**
         * Merges two accumulators and returns the result.
         *
         * @param acc1 The first accumulator to merge.
         * @param acc2 The second accumulator to merge.
         * @return The merged accumulator.
         */
        ACC merge(ACC acc1, ACC acc2);
    }
}
