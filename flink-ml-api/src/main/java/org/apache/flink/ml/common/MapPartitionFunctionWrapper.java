package org.apache.flink.ml.common;

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
