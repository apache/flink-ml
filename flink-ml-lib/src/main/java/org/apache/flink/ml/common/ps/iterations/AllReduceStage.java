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

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.util.function.SerializableSupplier;

import java.util.function.Supplier;

/**
 * This iteration stage is designed to perform an all-reduce operation on the specified array in a
 * distributed setting.
 *
 * <p>Users can specify how often this operation is conducted by setting the value of the
 * "executionInterval" parameter, which determines the frequency of the all-reduce stage. For
 * example, if the value of executionInterval is set to 5, the all-reduce stage will be executed
 * every 5 iterations.
 */
public final class AllReduceStage<V> implements IterationStage {
    public final Supplier<V[]> sendBuf;
    public final Supplier<V[]> recvBuf;
    public final ReduceFunction<V[]> reducer;
    public final TypeSerializer<V> typeSerializer;
    public final int executionInterval;

    public AllReduceStage(
            SerializableSupplier<V[]> sendBuf,
            SerializableSupplier<V[]> recvBuf,
            ReduceFunction<V[]> reducer,
            TypeSerializer<V> typeSerializer,
            int executionInterval) {
        this.sendBuf = sendBuf;
        this.recvBuf = recvBuf;
        this.reducer = reducer;
        this.typeSerializer = typeSerializer;
        this.executionInterval = executionInterval;
    }

    public AllReduceStage(
            SerializableSupplier<V[]> sendBuf,
            SerializableSupplier<V[]> recvBuf,
            ReduceFunction<V[]> reducer,
            TypeSerializer<V> typeSerializer) {
        this(sendBuf, recvBuf, reducer, typeSerializer, 1);
    }
}
