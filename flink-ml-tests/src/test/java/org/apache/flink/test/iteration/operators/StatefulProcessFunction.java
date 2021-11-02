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

package org.apache.flink.test.iteration.operators;

import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

/**
 * This is a function that uses keyed state so that we could verify the correctness of using keyed
 * stream inside the iteration.
 */
public class StatefulProcessFunction<T> extends KeyedProcessFunction<Integer, T, T> {

    private ValueState<Integer> state;

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        this.state =
                getRuntimeContext().getState(new ValueStateDescriptor<>("state", Integer.class));
    }

    @Override
    public void processElement(T value, Context ctx, Collector<T> out) throws Exception {
        if (state.value() == null) {
            state.update(0);

            // Trying registers a timer
            ctx.timerService().registerEventTimeTimer(1000L);
        } else {
            state.update(state.value() + 1);
        }

        out.collect(value);
    }
}
