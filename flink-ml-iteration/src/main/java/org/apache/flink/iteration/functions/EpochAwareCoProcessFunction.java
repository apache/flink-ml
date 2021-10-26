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

package org.apache.flink.iteration.functions;

import org.apache.flink.iteration.operator.allround.EpochAware;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.util.Collector;

import java.util.function.Supplier;

/**
 * A specialized {@link CoProcessFunction} that also provide the attach epoch with each record. It
 * is executed as all-round inside the iteration.
 */
public abstract class EpochAwareCoProcessFunction<I1, I2, O> extends CoProcessFunction<I1, I2, O>
        implements EpochAware {

    private Supplier<Integer> epochSupplier;

    @Override
    public void setEpochSupplier(Supplier<Integer> epochSupplier) {
        this.epochSupplier = epochSupplier;
    }

    @Override
    public final void processElement1(I1 input, Context context, Collector<O> collector)
            throws Exception {
        // For the sake of performance, we omit the check of nullability.
        processElement1(input, epochSupplier.get(), context, collector);
    }

    @Override
    public final void processElement2(I2 input, Context context, Collector<O> collector)
            throws Exception {
        // For the sake of performance, we omit the check of nullability.
        processElement2(input, epochSupplier.get(), context, collector);
    }

    public abstract void processElement1(
            I1 input, int epoch, Context context, Collector<O> collector) throws Exception;

    public abstract void processElement2(
            I2 input, int epoch, Context context, Collector<O> collector) throws Exception;
}
