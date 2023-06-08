package org.apache.flink.ml.common.ps.training;

import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

/** A collector that can only output using {@link OutputTag}. */
public final class ProxySideOutput {
    private final Output<?> output;

    public ProxySideOutput(Output<?> output) {
        this.output = output;
    }

    public <T> void output(OutputTag<T> outputTag, StreamRecord<T> record) {
        Preconditions.checkNotNull(outputTag);
        output.collect(outputTag, record);
    }
}
