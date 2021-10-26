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
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Collector;

/**
 * MapPartitionFunction wrapper.
 *
 * @param <IN> Input element type.
 * @param <OUT> Output element type.
 */
public class MapPartitionFunctionWrapper<IN, OUT> extends AbstractStreamOperator<OUT>
        implements OneInputStreamOperator<IN, OUT>, BoundedOneInput {
    private final ListStateDescriptor<IN> descriptor;
    private final MapPartitionFunction<IN, OUT> mapPartitionFunc;
    private ListState<IN> values;

    public MapPartitionFunctionWrapper(
            String uniqueName,
            TypeInformation<IN> typeInfo,
            MapPartitionFunction<IN, OUT> mapPartitionFunc) {
        this.descriptor = new ListStateDescriptor<>(uniqueName, typeInfo);
        this.mapPartitionFunc = mapPartitionFunc;
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        values = context.getOperatorStateStore().getListState(descriptor);
    }

    @Override
    public void endInput() throws Exception {
        Collector<OUT> out =
                new Collector<OUT>() {
                    @Override
                    public void collect(OUT value) {
                        output.collect(new StreamRecord<>(value));
                    }

                    @Override
                    public void close() {
                        output.close();
                    }
                };
        mapPartitionFunc.mapPartition(values.get(), out);
        values.clear();
    }

    @Override
    public void processElement(StreamRecord<IN> input) throws Exception {
        values.add(input.getValue());
    }
}
