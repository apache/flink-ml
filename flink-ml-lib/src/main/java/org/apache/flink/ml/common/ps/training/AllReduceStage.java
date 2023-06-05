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

import org.apache.flink.util.Preconditions;

import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Supplier;

/** A communication stage that conducts all-reduce on the given double array. */
public final class AllReduceStage implements IterationStage {
    public final Supplier<double[]> valuesSupplier;
    public final Consumer<double[]> valuesConsumer;
    public final BiFunction<double[], double[], double[]> valuesAggregator;

    public AllReduceStage(
            Supplier<double[]> valuesSupplier,
            Consumer<double[]> valuesConsumer,
            BiFunction<double[], double[], double[]> valuesAggregator) {
        this.valuesSupplier = valuesSupplier;
        this.valuesConsumer = valuesConsumer;
        this.valuesAggregator = valuesAggregator;
    }

    public AllReduceStage(Supplier<double[]> valuesSupplier, Consumer<double[]> valuesConsumer) {
        this(
                valuesSupplier,
                valuesConsumer,
                (SerializableBiFunction<double[], double[], double[]>)
                        (array1, array2) -> {
                            Preconditions.checkState(array1.length == array2.length);
                            for (int i = 0; i < array1.length; i++) {
                                array2[i] += array1[i];
                            }
                            return array2;
                        });
    }
}
