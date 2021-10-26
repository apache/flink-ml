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

package org.apache.flink.ml.common.datastream;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.runtime.state.KeyedStateFunction;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.VoidNamespace;
import org.apache.flink.runtime.state.VoidNamespaceSerializer;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

/**
 * Emit the last value of a bounded input stream.
 *
 * @param <T> Element type.
 */
public class ReduceFunctionWrapper<T> extends AbstractStreamOperator<T>
        implements OneInputStreamOperator<T, T>, BoundedOneInput {
    private final ValueStateDescriptor<T> descriptor;
    private final ReduceFunction<T> reduceFunc;
    private ValueState<T> lastRecord;

    public ReduceFunctionWrapper(
            String uniqueName, TypeInformation<T> typeInfo, ReduceFunction<T> reduceFunc) {
        this.descriptor = new ValueStateDescriptor<T>(uniqueName, typeInfo);
        this.reduceFunc = reduceFunc;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        lastRecord = context.getKeyedStateStore().getState(descriptor);
    }

    @Override
    public void endInput() throws Exception {
        KeyedStateFunction<Object, ValueState<T>> collectFunc =
                new KeyedStateFunction<Object, ValueState<T>>() {
                    @Override
                    public void process(Object object, ValueState<T> state) throws Exception {
                        output.collect(new StreamRecord<>(state.value()));
                        state.clear();
                    }
                };
        getKeyedStateBackend()
                .applyToAllKeys(
                        VoidNamespace.INSTANCE,
                        VoidNamespaceSerializer.INSTANCE,
                        descriptor,
                        collectFunc);
    }

    @Override
    public void processElement(StreamRecord<T> input) throws Exception {
        T newValue =
                lastRecord.value() == null
                        ? input.getValue()
                        : reduceFunc.reduce(lastRecord.value(), input.getValue());
        lastRecord.update(newValue);
    }
}
