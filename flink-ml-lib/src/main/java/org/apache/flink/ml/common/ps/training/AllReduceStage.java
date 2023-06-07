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

package org.apache.flink.ml.common.ps.training;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.typeutils.TypeSerializer;

import java.util.function.Consumer;
import java.util.function.Supplier;

/** A communication stage that conducts all-reduce on the given array. */
public final class AllReduceStage<V> implements IterationStage {
    public final Supplier<V[]> valuesSupplier;
    public final Consumer<V[]> valuesConsumer;
    public final ReduceFunction<V[]> reducer;
    public final TypeSerializer<V> typeSerializer;

    public AllReduceStage(
            Supplier<V[]> valuesSupplier,
            Consumer<V[]> valuesConsumer,
            ReduceFunction<V[]> reducer,
            TypeSerializer<V> typeSerializer) {
        this.valuesSupplier = valuesSupplier;
        this.valuesConsumer = valuesConsumer;
        this.reducer = reducer;
        this.typeSerializer = typeSerializer;
    }
}
