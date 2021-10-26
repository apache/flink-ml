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

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TimestampedCollector;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

/**
 * This utility class wraps a MapPartitionFunction into an OneInputStreamOperator so that a
 * MapPartitionFunction can be applied on a DataStream via the DataStream::transform API.
 *
 * @param <IN> The class type of the input element.
 * @param <OUT> The class type of the output element.
 */
public class MapPartitionFunctionWrapper<IN, OUT> extends AbstractStreamOperator<OUT>
        implements OneInputStreamOperator<IN, OUT>, BoundedOneInput {
    private final MapPartitionFunction<IN, OUT> mapPartitionFunc;
    private ListState<IN> values;

    public MapPartitionFunctionWrapper(MapPartitionFunction<IN, OUT> mapPartitionFunc) {
        this.mapPartitionFunc = mapPartitionFunc;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        ListStateDescriptor<IN> descriptor =
                new ListStateDescriptor<>(
                        "input",
                        getOperatorConfig().getTypeSerializerIn(0, getClass().getClassLoader()));
        values = context.getOperatorStateStore().getListState(descriptor);
    }

    @Override
    public void endInput() throws Exception {
        mapPartitionFunc.mapPartition(values.get(), new TimestampedCollector<>(output));
        values.clear();
    }

    @Override
    public void processElement(StreamRecord<IN> input) throws Exception {
        values.add(input.getValue());
    }
}
