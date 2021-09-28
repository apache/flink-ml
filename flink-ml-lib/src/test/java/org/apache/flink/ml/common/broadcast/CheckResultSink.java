package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;

import java.util.Iterator;

import static org.junit.Assert.assertEquals;

/**
 * The test sink that checks the size of the output. It will throw an exception if the number of
 * records received is not as expected.
 */
public class CheckResultSink extends RichSinkFunction<Long> implements CheckpointedFunction {
    private final int expectRecordsCnt;
    private int recordsReceivedCnt;

    private ListState<Integer> recordsReceivedCntState;

    public CheckResultSink(int expectRecordsCnt) {
        this.expectRecordsCnt = expectRecordsCnt;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
    }

    @Override
    public void invoke(Long value, Context context) {
        recordsReceivedCnt++;
    }

    @Override
    public void finish() {
        assertEquals(
                "Number of received records does not consistent",
                expectRecordsCnt,
                recordsReceivedCnt);
    }

    @Override
    public void close() {}

    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        this.recordsReceivedCntState.clear();
        this.recordsReceivedCntState.add(recordsReceivedCnt);
    }

    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext)
            throws Exception {
        recordsReceivedCntState =
                functionInitializationContext
                        .getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<Integer>(
                                        "recordsReceivedCnt", BasicTypeInfo.INT_TYPE_INFO));
        Iterator<Integer> iterator = recordsReceivedCntState.get().iterator();
        recordsReceivedCnt = 0;
        if (iterator.hasNext()) {
            recordsReceivedCnt = iterator.next();
        }
    }
}
