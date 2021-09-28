package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;

import java.util.Iterator;

public class TestSource extends RichParallelSourceFunction<Long> implements CheckpointedFunction {

    private static volatile boolean hasThrown = false;

    private ListState<Long> currentIdxState;
    private long currentIdx;
    private long mod, numPartitions, numPerPartition;
    private transient volatile boolean running = true;

    public TestSource(int numPerPartition) {
        this.numPerPartition = numPerPartition;
    }

    @Override
    public void open(Configuration parameters) {
        this.mod = getRuntimeContext().getIndexOfThisSubtask();
        this.numPartitions = getRuntimeContext().getNumberOfParallelSubtasks();
        running = true;
    }

    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        this.currentIdxState.clear();
        this.currentIdxState.add(currentIdx);
    }

    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext)
            throws Exception {
        currentIdxState =
                functionInitializationContext
                        .getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<Long>(
                                        "currentIdx", BasicTypeInfo.LONG_TYPE_INFO));
        Iterator<Long> iterator = currentIdxState.get().iterator();
        currentIdx = 0;
        if (iterator.hasNext()) {
            currentIdx = iterator.next();
        }
    }

    @Override
    public void run(SourceContext<Long> sourceContext) throws Exception {
        while (running && currentIdx < numPerPartition) {
            synchronized (sourceContext.getCheckpointLock()) {
                sourceContext.collect(currentIdx * numPartitions + mod);
                currentIdx++;
            }
            Thread.sleep(1);
            if (currentIdx == numPerPartition / 2 && (!hasThrown)) {
                hasThrown = true;
                throw new RuntimeException("Failing source");
            }
        }
    }

    @Override
    public void cancel() {
        running = false;
    }
}
