package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.ml.iteration.config.IterationOptions;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.MultipleConnectedStreams;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
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

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class BroadcastUtilsTest {
    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();
    private static int NUM_RECORDS_PER_PARTITION = 1000;
    private static int NUM_TM = 1;
    private static int NUM_SLOT = 1;
    private static int PARALLELISM = NUM_TM * NUM_SLOT;
    private static final String[] broadcastNames = new String[] {"source1", "source2"};

    private MiniClusterConfiguration createMiniClusterConfiguration(int numTm, int numSlot)
            throws IOException {
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18082);
        configuration.set(
                IterationOptions.DATA_CACHE_PATH,
                "file://" + tempFolder.newFolder().getAbsolutePath());
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        return new MiniClusterConfiguration.Builder()
                .setConfiguration(configuration)
                .setNumTaskManagers(numTm)
                .setNumSlotsPerTaskManager(numSlot)
                .build();
    }

    @Test
    public void testOneInputGraph() throws Exception {
        try (MiniCluster miniCluster =
                new MiniCluster(createMiniClusterConfiguration(NUM_TM, NUM_SLOT))) {
            miniCluster.start();
            JobGraph jobGraph = getJobGraph(1);
            miniCluster.executeJobBlocking(jobGraph);
        }
    }

    @Test
    public void testTwoInputGraph() throws Exception {
        try (MiniCluster miniCluster =
                new MiniCluster(createMiniClusterConfiguration(NUM_TM, NUM_SLOT))) {
            miniCluster.start();
            JobGraph jobGraph = getJobGraph(2);
            miniCluster.executeJobBlocking(jobGraph);
        }
    }

    @Test
    public void testMultiInputGraph() throws Exception {
        try (MiniCluster miniCluster =
                new MiniCluster(createMiniClusterConfiguration(NUM_TM, NUM_SLOT))) {
            miniCluster.start();
            JobGraph jobGraph = getJobGraph(3);
            miniCluster.executeJobBlocking(jobGraph);
        }
    }

    private JobGraph getJobGraph(int numNonBroadcastInputs) {
        StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment(
                        new Configuration() {
                            {
                                this.set(
                                        ExecutionCheckpointingOptions
                                                .ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH,
                                        true);
                            }
                        });
        env.enableCheckpointing(500, CheckpointingMode.EXACTLY_ONCE);
        env.setParallelism(NUM_SLOT * NUM_TM);

        DataStream<Long> source1 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        DataStream<Long> source2 = env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION));
        HashMap<String, DataStream<?>> bcStreamsMap = new HashMap<>();
        bcStreamsMap.put(broadcastNames[0], source1);
        bcStreamsMap.put(broadcastNames[1], source2);

        List<DataStream<?>> inputList = new ArrayList<>(1);
        // create a deadlock.
        inputList.add(source1);
        for (int i = 0; i < numNonBroadcastInputs - 1; i++) {
            inputList.add(env.addSource(new TestSource(NUM_RECORDS_PER_PARTITION)));
        }

        Function<List<DataStream<?>>, DataStream<Long>> func = getFunc(numNonBroadcastInputs);

        DataStream<Long> result =
                BroadcastUtils.<Long>withBroadcastStream(inputList, bcStreamsMap, func);

        result.addSink(
                        new CheckResultSink(
                                NUM_RECORDS_PER_PARTITION * PARALLELISM * numNonBroadcastInputs))
                .setParallelism(1);

        return env.getStreamGraph().getJobGraph();
    }

    private static Function<List<DataStream<?>>, DataStream<Long>> getFunc(int numInputs) {
        if (numInputs == 1) {
            return new Function<List<DataStream<?>>, DataStream<Long>>() {
                @Override
                public DataStream<Long> apply(List<DataStream<?>> dataStreams) {
                    DataStream input = dataStreams.get(0);
                    return input.transform(
                            "func", BasicTypeInfo.LONG_TYPE_INFO, new TestOneInputOp());
                }
            };
        } else if (numInputs == 2) {
            return new Function<List<DataStream<?>>, DataStream<Long>>() {
                @Override
                public DataStream<Long> apply(List<DataStream<?>> dataStreams) {
                    DataStream input1 = dataStreams.get(0);
                    DataStream input2 = dataStreams.get(1);
                    return input1.connect(input2)
                            .transform(
                                    "co-func", BasicTypeInfo.LONG_TYPE_INFO, new TestTwoInputOp());
                }
            };
        } else {
            return new Function<List<DataStream<?>>, DataStream<Long>>() {
                @Override
                public DataStream<Long> apply(List<DataStream<?>> dataStreams) {
                    StreamExecutionEnvironment env = dataStreams.get(0).getExecutionEnvironment();
                    MultipleInputTransformation<Long> multipleInputTransformation =
                            new MultipleInputTransformation<Long>(
                                    "multi-input",
                                    new TestMultiInputOpFactory(numInputs),
                                    BasicTypeInfo.LONG_TYPE_INFO,
                                    env.getParallelism());
                    for (int i = 0; i < dataStreams.size(); i++) {
                        multipleInputTransformation.addInput(
                                dataStreams.get(i).getTransformation());
                    }
                    env.addOperator(multipleInputTransformation);
                    return new MultipleConnectedStreams(env).transform(multipleInputTransformation);
                }
            };
        }
    }

    /**
     * The OneInputStreamOperator that checks the size of the broadcast inputs and pass the input to
     * downstream operator directly.
     */
    public static class TestOneInputOp extends AbstractStreamOperator<Long>
            implements OneInputStreamOperator<Long, Long> {
        @Override
        public void processElement(StreamRecord<Long> streamRecord) {
            List<Long> source1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
            List<Long> source2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
            assertEquals(NUM_RECORDS_PER_PARTITION * PARALLELISM, source1.size());
            assertEquals(NUM_RECORDS_PER_PARTITION * PARALLELISM, source2.size());

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
            List<Long> source1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
            List<Long> source2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
            assertEquals(NUM_RECORDS_PER_PARTITION * PARALLELISM, source1.size());
            assertEquals(NUM_RECORDS_PER_PARTITION * PARALLELISM, source2.size());
            output.collect(streamRecord);
        }

        @Override
        public void processElement2(StreamRecord<Long> streamRecord) throws Exception {
            List<Long> source1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
            List<Long> source2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
            assertEquals(NUM_RECORDS_PER_PARTITION * PARALLELISM, source1.size());
            assertEquals(NUM_RECORDS_PER_PARTITION * PARALLELISM, source2.size());
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
                inputList.add(new TestMultiInputOp.ProxyInput(this, i + 1));
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
            public void processElement(StreamRecord<Long> streamRecord) {
                List<Long> source1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
                List<Long> source2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
                assertEquals(source1.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
                assertEquals(source2.size(), NUM_RECORDS_PER_PARTITION * PARALLELISM);
                output.collect(streamRecord);
            }
        }
    }
}
