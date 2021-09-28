package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.contrib.streaming.state.EmbeddedRocksDBStateBackend;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.MultipleConnectedStreams;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;
import org.apache.flink.streaming.api.operators.AbstractInput;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorFactory;
import org.apache.flink.streaming.api.operators.AbstractStreamOperatorV2;
import org.apache.flink.streaming.api.operators.Input;
import org.apache.flink.streaming.api.operators.MultipleInputStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperatorParameters;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.api.transformations.MultipleInputTransformation;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class BroadcastUtilsTest {
    private static int NUM_RECORDS_PER_PARTITION = 100000;
    private static int PARALLELISM = 4;

    private StreamExecutionEnvironment getEnv(boolean enableCheckpoints) {
        Configuration conf = new Configuration();
        if (enableCheckpoints) {
            conf.setBoolean("execution.checkpointing.checkpoints-after-tasks-finish.enabled", true);
        }
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(conf);
        env.setParallelism(PARALLELISM);
        if (enableCheckpoints) {
            env.setStateBackend(new EmbeddedRocksDBStateBackend());
            env.getCheckpointConfig().setCheckpointStorage("file:///tmp/tmp_rocksDB_ckpt");
            env.enableCheckpointing(10000);
        }
        return env;
    }

    @Test
    public void testOneInput() throws Exception {
        StreamExecutionEnvironment env = getEnv(true);
        DataStream<Long> source1 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        DataStream<Long> source2 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        HashMap<String, DataStream<?>> bcStreamsMap = new HashMap<>();
        bcStreamsMap.put("source1", source1);
        bcStreamsMap.put("source2", source2);

        List<DataStream<?>> inputList = new ArrayList<>(1);
        inputList.add(source1);

        DataStream<Long> result =
                BroadcastUtils.<Long>withBroadcastStream(
                        inputList,
                        bcStreamsMap,
                        new Function<List<DataStream<?>>, DataStream<Long>>() {
                            @Override
                            public DataStream<Long> apply(List<DataStream<?>> dataStreams) {
                                DataStream input = dataStreams.get(0);
                                return input.transform(
                                        "func", BasicTypeInfo.LONG_TYPE_INFO, new TestOneInputOp());
                            }
                        },
                        BasicTypeInfo.LONG_TYPE_INFO);

        result.addSink(new CheckResultSink(NUM_RECORDS_PER_PARTITION * PARALLELISM))
                .setParallelism(1);
        env.execute();
    }

    @Test
    public void testTwoInput() throws Exception {
        StreamExecutionEnvironment env = getEnv(true);
        DataStream<Long> source1 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        DataStream<Long> source2 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        HashMap<String, DataStream<?>> bcStreamsMap = new HashMap<>();
        bcStreamsMap.put("source1", source1);
        bcStreamsMap.put("source2", source2);

        List<DataStream<?>> inputList = new ArrayList<>(2);
        inputList.add(source1);
        inputList.add(source2);

        DataStream<Long> result =
                BroadcastUtils.<Long>withBroadcastStream(
                        inputList,
                        bcStreamsMap,
                        new Function<List<DataStream<?>>, DataStream<Long>>() {
                            @Override
                            public DataStream<Long> apply(List<DataStream<?>> dataStreams) {
                                DataStream input1 = dataStreams.get(0);
                                DataStream input2 = dataStreams.get(1);
                                return input1.connect(input2)
                                        .transform(
                                                "co-func",
                                                BasicTypeInfo.LONG_TYPE_INFO,
                                                new TestTwoInputOp());
                            }
                        },
                        BasicTypeInfo.LONG_TYPE_INFO);

        result.addSink(new CheckResultSink(NUM_RECORDS_PER_PARTITION * PARALLELISM * 2))
                .setParallelism(1);
        env.execute();
    }

    @Test
    public void testMultiInput() throws Exception {
        StreamExecutionEnvironment env = getEnv(true);
        DataStream<Long> source1 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        DataStream<Long> source2 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        HashMap<String, DataStream<?>> bcStreamsMap = new HashMap<>();
        bcStreamsMap.put("source1", source1);
        bcStreamsMap.put("source2", source2);

        List<DataStream<?>> inputList = new ArrayList<>(1);
        inputList.add(source1);
        inputList.add(source2);

        DataStream<Long> result =
                BroadcastUtils.<Long>withBroadcastStream(
                        inputList,
                        bcStreamsMap,
                        new Function<List<DataStream<?>>, DataStream<Long>>() {
                            @Override
                            public DataStream<Long> apply(List<DataStream<?>> dataStreams) {
                                DataStream input1 = dataStreams.get(0);
                                DataStream input2 = dataStreams.get(1);
                                MultipleInputTransformation<Long> multipleInputTransformation =
                                        new MultipleInputTransformation<Long>(
                                                "multi-input",
                                                new TestMultiInputOpFactory(2),
                                                BasicTypeInfo.LONG_TYPE_INFO,
                                                input1.getParallelism());
                                multipleInputTransformation.addInput(input1.getTransformation());
                                multipleInputTransformation.addInput(input2.getTransformation());
                                input1.getExecutionEnvironment()
                                        .addOperator(multipleInputTransformation);
                                return new MultipleConnectedStreams(
                                                input1.getExecutionEnvironment())
                                        .transform(multipleInputTransformation);
                            }
                        },
                        BasicTypeInfo.LONG_TYPE_INFO);

        result.addSink(new CheckResultSink(NUM_RECORDS_PER_PARTITION * PARALLELISM * 2))
                .setParallelism(1);
        env.execute();
    }

    // ------------------------------------------------------------------------
    //  Utilities
    // ------------------------------------------------------------------------
    private static class TestSource extends RichParallelSourceFunction<Long>
            implements CheckpointedFunction {

        private ListState<Long> checkpointOffset;
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
            currentIdx = 0;
            running = true;
        }

        @Override
        public void snapshotState(FunctionSnapshotContext functionSnapshotContext)
                throws Exception {
            this.checkpointOffset.clear();
            this.checkpointOffset.add(currentIdx);
        }

        @Override
        public void initializeState(FunctionInitializationContext functionInitializationContext)
                throws Exception {
            checkpointOffset =
                    functionInitializationContext
                            .getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<Long>(
                                            "offset", BasicTypeInfo.LONG_TYPE_INFO));
            Iterator<Long> iterator = checkpointOffset.get().iterator();
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
                if (currentIdx % 500 == 0) {
                    Thread.sleep(10);
                }
            }
        }

        @Override
        public void cancel() {
            running = false;
        }
    }

    /**
     * The OneInputStreamOperator that checks the size of the broadcast inputs and pass the input to
     * downstream operator directly.
     */
    private static class TestOneInputOp extends AbstractStreamOperator<Long>
            implements OneInputStreamOperator<Long, Long> {
        @Override
        public void processElement(StreamRecord<Long> streamRecord) throws Exception {
            List<Long> source1 = BroadcastContext.getBroadcastVariable("source1");
            List<Long> source2 = BroadcastContext.getBroadcastVariable("source2");
            assertEquals(source1.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
            assertEquals(source2.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
            output.collect(streamRecord);
        }
    }

    /**
     * The TwoInputStreamOperator that check the size of the broadcast inputs and pass the input to
     * downstream operator directly.
     */
    private static class TestTwoInputOp extends AbstractStreamOperator<Long>
            implements TwoInputStreamOperator<Long, Long, Long> {
        @Override
        public void processElement1(StreamRecord<Long> streamRecord) throws Exception {
            List<Long> source1 = BroadcastContext.getBroadcastVariable("source1");
            List<Long> source2 = BroadcastContext.getBroadcastVariable("source2");
            assertEquals(source1.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
            assertEquals(source2.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
            output.collect(streamRecord);
        }

        @Override
        public void processElement2(StreamRecord<Long> streamRecord) throws Exception {
            List<Long> source1 = BroadcastContext.getBroadcastVariable("source1");
            List<Long> source2 = BroadcastContext.getBroadcastVariable("source2");
            assertEquals(source1.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
            assertEquals(source2.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
            output.collect(streamRecord);
        }
    }

    /** Factory class for {@link TestMultiInputOp}. */
    private static class TestMultiInputOpFactory extends AbstractStreamOperatorFactory<Long> {
        private int numInputs;

        public TestMultiInputOpFactory(int numInputs) {
            this.numInputs = numInputs;
        }

        @Override
        public <T extends StreamOperator<Long>> T createStreamOperator(
                StreamOperatorParameters<Long> streamOperatorParameters) {
            return (T) new TestMultiInputOp(streamOperatorParameters, numInputs);
        }

        @Override
        public Class<? extends StreamOperator> getStreamOperatorClass(ClassLoader classLoader) {
            return TestMultiInputOp.class;
        }
    }

    /**
     * The MultiInputStreamOperator that checks the size of the broadcast inputs and pass the input
     * to downstream operator directly.
     */
    private static class TestMultiInputOp extends AbstractStreamOperatorV2<Long>
            implements MultipleInputStreamOperator<Long> {
        private List<Input> inputList;

        public TestMultiInputOp(StreamOperatorParameters<Long> parameters, int numberOfInputs) {
            super(parameters, numberOfInputs);
            this.inputList = new ArrayList<>(numberOfInputs);
            for (int i = 0; i < numberOfInputs; i++) {
                inputList.add(new ProxyInput(this, i + 1));
            }
        }

        @Override
        public List<Input> getInputs() {
            return inputList;
        }

        private class ProxyInput extends AbstractInput<Long, Long> {

            public ProxyInput(AbstractStreamOperatorV2<Long> owner, int inputId) {
                super(owner, inputId);
            }

            @Override
            public void processElement(StreamRecord<Long> streamRecord) throws Exception {
                List<Long> source1 = BroadcastContext.getBroadcastVariable("source1");
                List<Long> source2 = BroadcastContext.getBroadcastVariable("source2");
                assertEquals(source1.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
                assertEquals(source2.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
                output.collect(streamRecord);
            }
        }
    }

    /**
     * The test sink that checks the size of the output. It will throw an exception if the number of
     * records received is not as expected.
     */
    private static class CheckResultSink extends RichSinkFunction<Long> {
        private int recordsReceivedCnt;
        private final int expectRecordsCnt;

        public CheckResultSink(int expectRecordsCnt) {
            this.expectRecordsCnt = expectRecordsCnt;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            recordsReceivedCnt = 0;
        }

        @Override
        public void invoke(Long value, Context context) {
            recordsReceivedCnt++;
        }

        @Override
        public void close() {
            assertEquals(
                    "Number of received records does not consistent",
                    expectRecordsCnt,
                    recordsReceivedCnt);
        }
    }
}
