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

package org.apache.flink.ml.classification;

import org.apache.flink.api.common.JobID;
import org.apache.flink.api.common.JobSubmissionResult;
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelDataSegment;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelDataUtil;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegression;
import org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.InMemorySinkFunction;
import org.apache.flink.ml.util.InMemorySourceFunction;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.runtime.client.JobStatusMessage;
import org.apache.flink.runtime.jobgraph.JobGraph;
import org.apache.flink.runtime.minicluster.MiniCluster;
import org.apache.flink.runtime.minicluster.MiniClusterConfiguration;
import org.apache.flink.runtime.testutils.InMemoryReporter;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.TestLogger;

import org.apache.commons.collections.IteratorUtils;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.apache.flink.ml.classification.logisticregression.OnlineLogisticRegressionModel.MODEL_DATA_VERSION_GAUGE_KEY;

/** Tests {@link OnlineLogisticRegression} and {@link OnlineLogisticRegressionModel}. */
public class OnlineLogisticRegressionTest extends TestLogger {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private static final double[] ONE_ARRAY = new double[] {1.0, 1.0, 1.0};

    private static final Row[] TRAIN_DENSE_ROWS_1 =
            new Row[] {
                Row.of(Vectors.dense(0.1, 2.), 0.),
                Row.of(Vectors.dense(0.2, 2.), 0.),
                Row.of(Vectors.dense(0.3, 2.), 0.),
                Row.of(Vectors.dense(0.4, 2.), 0.),
                Row.of(Vectors.dense(0.5, 2.), 0.),
                Row.of(Vectors.dense(11., 12.), 1.),
                Row.of(Vectors.dense(12., 11.), 1.),
                Row.of(Vectors.dense(13., 12.), 1.),
                Row.of(Vectors.dense(14., 12.), 1.),
                Row.of(Vectors.dense(15., 12.), 1.)
            };

    private static final Row[] TRAIN_DENSE_ROWS_2 =
            new Row[] {
                Row.of(Vectors.dense(0.2, 3.), 0.),
                Row.of(Vectors.dense(0.8, 1.), 0.),
                Row.of(Vectors.dense(0.7, 1.), 0.),
                Row.of(Vectors.dense(0.6, 2.), 0.),
                Row.of(Vectors.dense(0.2, 2.), 0.),
                Row.of(Vectors.dense(14., 17.), 1.),
                Row.of(Vectors.dense(15., 10.), 1.),
                Row.of(Vectors.dense(16., 16.), 1.),
                Row.of(Vectors.dense(17., 10.), 1.),
                Row.of(Vectors.dense(18., 13.), 1.)
            };

    private static final Row[] PREDICT_DENSE_ROWS =
            new Row[] {
                Row.of(Vectors.dense(0.8, 2.7), 0.0), Row.of(Vectors.dense(15.5, 11.2), 1.0)
            };

    private static final Row[] TRAIN_SPARSE_ROWS_1 =
            new Row[] {
                Row.of(Vectors.sparse(10, new int[] {1, 3, 4}, ONE_ARRAY), 0., 1.0),
                Row.of(Vectors.sparse(10, new int[] {0, 2, 3}, ONE_ARRAY), 0., 1.4),
                Row.of(Vectors.sparse(10, new int[] {0, 3, 4}, ONE_ARRAY), 0., 1.3),
                Row.of(Vectors.sparse(10, new int[] {2, 3, 4}, ONE_ARRAY), 0., 1.4),
                Row.of(Vectors.sparse(10, new int[] {1, 3, 4}, ONE_ARRAY), 0., 1.6),
                Row.of(Vectors.sparse(10, new int[] {6, 7, 8}, ONE_ARRAY), 1., 1.8),
                Row.of(Vectors.sparse(10, new int[] {6, 8, 9}, ONE_ARRAY), 1., 1.9),
                Row.of(Vectors.sparse(10, new int[] {5, 8, 9}, ONE_ARRAY), 1., 1.0),
                Row.of(Vectors.sparse(10, new int[] {5, 6, 7}, ONE_ARRAY), 1., 1.1)
            };

    private static final Row[] TRAIN_SPARSE_ROWS_2 =
            new Row[] {
                Row.of(Vectors.sparse(10, new int[] {1, 2, 4}, ONE_ARRAY), 0., 1.0),
                Row.of(Vectors.sparse(10, new int[] {2, 3, 4}, ONE_ARRAY), 0., 1.3),
                Row.of(Vectors.sparse(10, new int[] {0, 2, 4}, ONE_ARRAY), 0., 1.4),
                Row.of(Vectors.sparse(10, new int[] {1, 3, 4}, ONE_ARRAY), 0., 1.0),
                Row.of(Vectors.sparse(10, new int[] {6, 7, 9}, ONE_ARRAY), 1., 1.6),
                Row.of(Vectors.sparse(10, new int[] {7, 8, 9}, ONE_ARRAY), 1., 1.8),
                Row.of(Vectors.sparse(10, new int[] {5, 7, 9}, ONE_ARRAY), 1., 1.0),
                Row.of(Vectors.sparse(10, new int[] {5, 6, 7}, ONE_ARRAY), 1., 1.5),
                Row.of(Vectors.sparse(10, new int[] {5, 8, 9}, ONE_ARRAY), 1., 1.0)
            };

    private static final Row[] PREDICT_SPARSE_ROWS =
            new Row[] {
                Row.of(Vectors.sparse(10, new int[] {1, 3, 5}, ONE_ARRAY), 0.),
                Row.of(Vectors.sparse(10, new int[] {5, 8, 9}, ONE_ARRAY), 1.)
            };

    private static final int defaultParallelism = 4;
    private static final int numTaskManagers = 2;
    private static final int numSlotsPerTaskManager = 2;

    private long currentModelDataVersion;

    private InMemorySourceFunction<Row> trainDenseSource;
    private InMemorySourceFunction<Row> predictDenseSource;
    private InMemorySourceFunction<Row> trainSparseSource;
    private InMemorySourceFunction<Row> predictSparseSource;
    private InMemorySinkFunction<Row> outputSink;
    private InMemorySinkFunction<LogisticRegressionModelDataSegment> modelDataSink;

    private static InMemoryReporter reporter;
    private static MiniCluster miniCluster;
    private static StreamExecutionEnvironment env;
    private static StreamTableEnvironment tEnv;

    private Table offlineTrainDenseTable;
    private Table onlineTrainDenseTable;
    private Table onlinePredictDenseTable;
    private Table onlineTrainSparseTable;
    private Table onlinePredictSparseTable;
    private Table initDenseModel;
    private Table initSparseModel;

    @BeforeClass
    public static void beforeClass() throws Exception {
        Configuration config = new Configuration();
        config.set(RestOptions.BIND_PORT, "18081-19091");
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        reporter = InMemoryReporter.create();
        reporter.addToConfiguration(config);

        miniCluster =
                new MiniCluster(
                        new MiniClusterConfiguration.Builder()
                                .setConfiguration(config)
                                .setNumTaskManagers(numTaskManagers)
                                .setNumSlotsPerTaskManager(numSlotsPerTaskManager)
                                .build());
        miniCluster.start();

        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setParallelism(defaultParallelism);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
    }

    @Before
    public void before() throws Exception {
        currentModelDataVersion = 0;

        trainDenseSource = new InMemorySourceFunction<>();
        predictDenseSource = new InMemorySourceFunction<>();
        trainSparseSource = new InMemorySourceFunction<>();
        predictSparseSource = new InMemorySourceFunction<>();
        outputSink = new InMemorySinkFunction<>();
        modelDataSink = new InMemorySinkFunction<>();

        offlineTrainDenseTable =
                tEnv.fromDataStream(env.fromElements(TRAIN_DENSE_ROWS_1)).as("features", "label");
        onlineTrainDenseTable =
                tEnv.fromDataStream(
                        env.addSource(
                                trainDenseSource,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(DenseIntDoubleVector.class),
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label"})));

        onlinePredictDenseTable =
                tEnv.fromDataStream(
                        env.addSource(
                                predictDenseSource,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(DenseIntDoubleVector.class),
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label"})));

        onlineTrainSparseTable =
                tEnv.fromDataStream(
                        env.addSource(
                                trainSparseSource,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(SparseIntDoubleVector.class),
                                            Types.DOUBLE,
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label", "weight"})));

        onlinePredictSparseTable =
                tEnv.fromDataStream(
                        env.addSource(
                                predictSparseSource,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            TypeInformation.of(SparseIntDoubleVector.class),
                                            Types.DOUBLE
                                        },
                                        new String[] {"features", "label"})));

        initDenseModel =
                tEnv.fromDataStream(
                        env.fromElements(
                                Row.of(
                                        new DenseIntDoubleVector(
                                                new double[] {
                                                    0.41233679404769874, -0.18088118293232122
                                                }),
                                        0L,
                                        2L,
                                        0L)));
        initSparseModel =
                tEnv.fromDataStream(
                        env.fromElements(
                                Row.of(
                                        new DenseIntDoubleVector(
                                                new double[] {
                                                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                                                    0.01, 0.01
                                                }),
                                        0L,
                                        10L,
                                        0L)));
    }

    @After
    public void after() throws Exception {
        for (JobStatusMessage message : miniCluster.listJobs().get()) {
            miniCluster.cancelJob(message.getJobId());
        }
    }

    @AfterClass
    public static void afterClass() throws Exception {
        miniCluster.close();
    }

    /**
     * Performs transform() on the provided model with predictTable, and adds sinks for
     * OnlineLogisticRegressionModel's transform output and model data.
     */
    private void transformAndOutputData(
            OnlineLogisticRegressionModel onlineModel, boolean isSparse) {
        Table outputTable =
                onlineModel
                        .transform(isSparse ? onlinePredictSparseTable : onlinePredictDenseTable)[
                        0];
        tEnv.toDataStream(outputTable).addSink(outputSink);

        Table modelDataTable = onlineModel.getModelData()[0];
        LogisticRegressionModelDataUtil.getModelDataStream(modelDataTable).addSink(modelDataSink);
    }

    /** Blocks the thread until Model has set up init model data. */
    private void waitInitModelDataSetup(JobID jobID) throws InterruptedException {
        while (reporter.findMetrics(jobID, MODEL_DATA_VERSION_GAUGE_KEY).size()
                < defaultParallelism) {
            Thread.sleep(100);
        }
        waitModelDataUpdate(jobID);
    }

    /** Blocks the thread until the Model has received the next model-data-update event. */
    @SuppressWarnings("unchecked")
    private void waitModelDataUpdate(JobID jobID) throws InterruptedException {
        do {
            long tmpModelDataVersion =
                    reporter.findMetrics(jobID, MODEL_DATA_VERSION_GAUGE_KEY).values().stream()
                            .map(x -> Long.parseLong(((Gauge<String>) x).getValue()))
                            .min(Long::compareTo)
                            .get();
            if (tmpModelDataVersion == currentModelDataVersion) {
                Thread.sleep(100);
            } else {
                currentModelDataVersion = tmpModelDataVersion;
                break;
            }
        } while (true);
    }

    /**
     * Inserts default predict data to the predict queue, fetches the prediction results, and
     * asserts that the grouping result is as expected.
     *
     * @param expectedRawInfo A list containing sets of expected result RawInfo.
     */
    private void predictAndAssert(List<DenseIntDoubleVector> expectedRawInfo, boolean isSparse)
            throws Exception {
        if (isSparse) {
            predictSparseSource.addAll(PREDICT_SPARSE_ROWS);
        } else {
            predictDenseSource.addAll(PREDICT_DENSE_ROWS);
        }
        List<Row> rawResult =
                outputSink.poll(isSparse ? PREDICT_SPARSE_ROWS.length : PREDICT_DENSE_ROWS.length);
        List<DenseIntDoubleVector> resultDetail = new ArrayList<>(rawResult.size());
        for (Row row : rawResult) {
            resultDetail.add(row.getFieldAs(3));
        }
        resultDetail.sort(TestUtils::compare);
        expectedRawInfo.sort(TestUtils::compare);
        for (int i = 0; i < resultDetail.size(); ++i) {
            double[] realData = resultDetail.get(i).values;
            double[] expectedData = expectedRawInfo.get(i).values;
            for (int j = 0; j < expectedData.length; ++j) {
                Assert.assertEquals(realData[j], expectedData[j], 1.0e-5);
            }
        }
    }

    private JobID submitJob(JobGraph jobGraph)
            throws ExecutionException, InterruptedException, TimeoutException {
        return miniCluster
                .submitJob(jobGraph)
                .thenApply(JobSubmissionResult::getJobID)
                .get(1, TimeUnit.SECONDS);
    }

    @Test
    public void testParam() {
        OnlineLogisticRegression onlineLogisticRegression = new OnlineLogisticRegression();
        Assert.assertEquals("features", onlineLogisticRegression.getFeaturesCol());
        Assert.assertEquals("count", onlineLogisticRegression.getBatchStrategy());
        Assert.assertEquals("label", onlineLogisticRegression.getLabelCol());
        Assert.assertEquals(0.0, onlineLogisticRegression.getReg(), 1.0e-5);
        Assert.assertEquals(0.0, onlineLogisticRegression.getElasticNet(), 1.0e-5);
        Assert.assertEquals(0.1, onlineLogisticRegression.getAlpha(), 1.0e-5);
        Assert.assertEquals(0.1, onlineLogisticRegression.getBeta(), 1.0e-5);
        Assert.assertEquals(32, onlineLogisticRegression.getGlobalBatchSize());

        onlineLogisticRegression
                .setFeaturesCol("test_feature")
                .setLabelCol("test_label")
                .setGlobalBatchSize(5)
                .setReg(0.5)
                .setElasticNet(0.25)
                .setAlpha(0.1)
                .setBeta(0.2);

        Assert.assertEquals("test_feature", onlineLogisticRegression.getFeaturesCol());
        Assert.assertEquals("test_label", onlineLogisticRegression.getLabelCol());
        Assert.assertEquals(0.5, onlineLogisticRegression.getReg(), 1.0e-5);
        Assert.assertEquals(0.25, onlineLogisticRegression.getElasticNet(), 1.0e-5);
        Assert.assertEquals(0.1, onlineLogisticRegression.getAlpha(), 1.0e-5);
        Assert.assertEquals(0.2, onlineLogisticRegression.getBeta(), 1.0e-5);
        Assert.assertEquals(5, onlineLogisticRegression.getGlobalBatchSize());

        OnlineLogisticRegressionModel onlineLogisticRegressionModel =
                new OnlineLogisticRegressionModel();
        Assert.assertEquals("features", onlineLogisticRegressionModel.getFeaturesCol());
        Assert.assertEquals("modelVersion", onlineLogisticRegressionModel.getModelVersionCol());
        Assert.assertEquals("prediction", onlineLogisticRegressionModel.getPredictionCol());
        Assert.assertEquals("rawPrediction", onlineLogisticRegressionModel.getRawPredictionCol());

        onlineLogisticRegressionModel
                .setFeaturesCol("test_feature")
                .setPredictionCol("pred")
                .setModelVersionCol("version")
                .setRawPredictionCol("raw");

        Assert.assertEquals("test_feature", onlineLogisticRegressionModel.getFeaturesCol());
        Assert.assertEquals("version", onlineLogisticRegressionModel.getModelVersionCol());
        Assert.assertEquals("pred", onlineLogisticRegressionModel.getPredictionCol());
        Assert.assertEquals("raw", onlineLogisticRegressionModel.getRawPredictionCol());
    }

    @Test
    public void testDenseFitAndPredict() throws Exception {
        final List<DenseIntDoubleVector> expectedRawInfo1 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.04481034155642882, 0.9551896584435712}),
                        new DenseIntDoubleVector(
                                new double[] {0.5353966697318491, 0.4646033302681509}));
        final List<DenseIntDoubleVector> expectedRawInfo2 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.013104324065967066, 0.9868956759340329}),
                        new DenseIntDoubleVector(
                                new double[] {0.5095144380001769, 0.49048556199982307}));
        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(initDenseModel);
        OnlineLogisticRegressionModel onlineModel =
                onlineLogisticRegression.fit(onlineTrainDenseTable);
        transformAndOutputData(onlineModel, false);

        final JobID jobID = submitJob(env.getStreamGraph().getJobGraph());
        trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);

        waitInitModelDataSetup(jobID);
        predictAndAssert(expectedRawInfo1, false);

        trainDenseSource.addAll(TRAIN_DENSE_ROWS_2);
        waitModelDataUpdate(jobID);
        predictAndAssert(expectedRawInfo2, false);
    }

    @Test
    public void testSparseFitAndPredict() throws Exception {
        final List<DenseIntDoubleVector> expectedRawInfo1 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.4452309884735286, 0.5547690115264714}),
                        new DenseIntDoubleVector(
                                new double[] {0.5105551725414953, 0.4894448274585047}));
        final List<DenseIntDoubleVector> expectedRawInfo2 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.40310431554310666, 0.5968956844568933}),
                        new DenseIntDoubleVector(
                                new double[] {0.5249618837373886, 0.4750381162626114}));
        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(9)
                        .setInitialModelData(initSparseModel);
        OnlineLogisticRegressionModel onlineModel =
                onlineLogisticRegression.fit(onlineTrainSparseTable);
        transformAndOutputData(onlineModel, true);
        final JobID jobID = submitJob(env.getStreamGraph().getJobGraph());

        trainSparseSource.addAll(TRAIN_SPARSE_ROWS_1);
        waitInitModelDataSetup(jobID);
        predictAndAssert(expectedRawInfo1, true);

        trainSparseSource.addAll(TRAIN_SPARSE_ROWS_2);
        waitModelDataUpdate(jobID);
        predictAndAssert(expectedRawInfo2, true);
    }

    @Test
    public void testFitAndPredictWithWeightCol() throws Exception {
        final List<DenseIntDoubleVector> expectedRawInfo1 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.452491993753382, 0.547508006246618}),
                        new DenseIntDoubleVector(
                                new double[] {0.5069192929506545, 0.4930807070493455}));
        final List<DenseIntDoubleVector> expectedRawInfo2 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.41108882806164193, 0.5889111719383581}),
                        new DenseIntDoubleVector(
                                new double[] {0.5247727600974581, 0.4752272399025419}));
        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setWeightCol("weight")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(9)
                        .setInitialModelData(initSparseModel);
        OnlineLogisticRegressionModel onlineModel =
                onlineLogisticRegression.fit(onlineTrainSparseTable);
        transformAndOutputData(onlineModel, true);
        final JobID jobID = submitJob(env.getStreamGraph().getJobGraph());

        trainSparseSource.addAll(TRAIN_SPARSE_ROWS_1);
        waitInitModelDataSetup(jobID);
        predictAndAssert(expectedRawInfo1, true);

        trainSparseSource.addAll(TRAIN_SPARSE_ROWS_2);
        waitModelDataUpdate(jobID);
        predictAndAssert(expectedRawInfo2, true);
    }

    @Test
    public void testGenerateRandomModelData() throws Exception {
        Table modelDataTable =
                LogisticRegressionModelDataUtil.generateRandomModelData(tEnv, 2, 2022);
        DataStream<Row> modelData = tEnv.toDataStream(modelDataTable);
        Row modelRow = (Row) IteratorUtils.toList(modelData.executeAndCollect()).get(0);
        Assert.assertEquals(2, ((DenseIntDoubleVector) modelRow.getField(0)).size().intValue());
        Assert.assertEquals(0L, modelRow.getField(1));
    }

    @Test
    public void testInitWithLogisticRegression() throws Exception {
        final List<DenseIntDoubleVector> expectedRawInfo1 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.037327343811250024, 0.96267265618875}),
                        new DenseIntDoubleVector(
                                new double[] {0.5684728224189707, 0.4315271775810293}));
        final List<DenseIntDoubleVector> expectedRawInfo2 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.007758574555505882, 0.9922414254444941}),
                        new DenseIntDoubleVector(
                                new double[] {0.5257216567388069, 0.4742783432611931}));
        LogisticRegression logisticRegression =
                new LogisticRegression()
                        .setLabelCol("label")
                        .setFeaturesCol("features")
                        .setPredictionCol("prediction");
        LogisticRegressionModel model = logisticRegression.fit(offlineTrainDenseTable);

        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(model.getModelData()[0]);

        OnlineLogisticRegressionModel onlineModel =
                onlineLogisticRegression.fit(onlineTrainDenseTable);
        transformAndOutputData(onlineModel, false);
        final JobID jobID = submitJob(env.getStreamGraph().getJobGraph());
        trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);
        waitInitModelDataSetup(jobID);
        predictAndAssert(expectedRawInfo1, false);

        trainDenseSource.addAll(TRAIN_DENSE_ROWS_2);
        waitModelDataUpdate(jobID);
        predictAndAssert(expectedRawInfo2, false);
    }

    @Test
    public void testBatchSizeLessThanParallelism() {
        try {
            new OnlineLogisticRegression()
                    .setInitialModelData(initDenseModel)
                    .setReg(0.2)
                    .setElasticNet(0.5)
                    .setGlobalBatchSize(2)
                    .setLabelCol("label")
                    .fit(onlineTrainDenseTable);
            Assert.fail("Expected IllegalStateException");
        } catch (Exception e) {
            Throwable exception = e;
            while (exception.getCause() != null) {
                exception = exception.getCause();
            }
            Assert.assertEquals(IllegalStateException.class, exception.getClass());
            Assert.assertEquals(
                    "There are more subtasks in the training process than the number "
                            + "of elements in each batch. Some subtasks might be idling forever.",
                    exception.getMessage());
        }
    }

    @Test
    public void testSaveAndReload() throws Exception {
        final List<DenseIntDoubleVector> expectedRawInfo1 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.04481034155642882, 0.9551896584435712}),
                        new DenseIntDoubleVector(
                                new double[] {0.5353966697318491, 0.4646033302681509}));
        final List<DenseIntDoubleVector> expectedRawInfo2 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.013104324065967066, 0.9868956759340329}),
                        new DenseIntDoubleVector(
                                new double[] {0.5095144380001769, 0.49048556199982307}));
        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(initDenseModel);

        String savePath = tempFolder.newFolder().getAbsolutePath();
        onlineLogisticRegression.save(savePath);
        miniCluster.executeJobBlocking(env.getStreamGraph().getJobGraph());
        OnlineLogisticRegression loadedOnlineLogisticRegression =
                OnlineLogisticRegression.load(tEnv, savePath);
        OnlineLogisticRegressionModel onlineModel =
                loadedOnlineLogisticRegression.fit(onlineTrainDenseTable);
        String modelSavePath = tempFolder.newFolder().getAbsolutePath();
        onlineModel.save(modelSavePath);
        OnlineLogisticRegressionModel loadedOnlineModel =
                OnlineLogisticRegressionModel.load(tEnv, modelSavePath);
        loadedOnlineModel.setModelData(onlineModel.getModelData());

        transformAndOutputData(loadedOnlineModel, false);
        final JobID jobID = submitJob(env.getStreamGraph().getJobGraph());

        trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);
        waitInitModelDataSetup(jobID);
        predictAndAssert(expectedRawInfo1, false);

        trainDenseSource.addAll(TRAIN_DENSE_ROWS_2);
        waitModelDataUpdate(jobID);
        predictAndAssert(expectedRawInfo2, false);
    }

    @Test
    public void testGetModelData() throws Exception {
        OnlineLogisticRegression onlineLogisticRegression =
                new OnlineLogisticRegression()
                        .setFeaturesCol("features")
                        .setPredictionCol("prediction")
                        .setReg(0.2)
                        .setElasticNet(0.5)
                        .setGlobalBatchSize(10)
                        .setInitialModelData(initDenseModel);
        OnlineLogisticRegressionModel onlineModel =
                onlineLogisticRegression.fit(onlineTrainDenseTable);
        transformAndOutputData(onlineModel, false);

        submitJob(env.getStreamGraph().getJobGraph());
        trainDenseSource.addAll(TRAIN_DENSE_ROWS_1);
        LogisticRegressionModelDataSegment actualModelData = modelDataSink.poll();

        LogisticRegressionModelDataSegment expectedModelData =
                new LogisticRegressionModelDataSegment(
                        new DenseIntDoubleVector(
                                new double[] {0.2994527071464283, -0.1412541067743284}),
                        1L);
        Assert.assertArrayEquals(
                expectedModelData.coefficient.values, actualModelData.coefficient.values, 1e-5);
        Assert.assertEquals(expectedModelData.modelVersion, actualModelData.modelVersion);
    }

    @Test
    public void testSetModelData() throws Exception {
        LogisticRegressionModelDataSegment modelData1 =
                new LogisticRegressionModelDataSegment(
                        new DenseIntDoubleVector(new double[] {0.085, -0.22}), 1L);

        LogisticRegressionModelDataSegment modelData2 =
                new LogisticRegressionModelDataSegment(
                        new DenseIntDoubleVector(new double[] {0.075, -0.28}), 2L);

        final List<DenseIntDoubleVector> expectedRawInfo1 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.6285496932692606, 0.3714503067307394}),
                        new DenseIntDoubleVector(
                                new double[] {0.7588710471221473, 0.24112895287785274}));
        final List<DenseIntDoubleVector> expectedRawInfo2 =
                Arrays.asList(
                        new DenseIntDoubleVector(
                                new double[] {0.6673003248270917, 0.3326996751729083}),
                        new DenseIntDoubleVector(
                                new double[] {0.8779865510655934, 0.12201344893440658}));

        InMemorySourceFunction<LogisticRegressionModelDataSegment> modelDataSource =
                new InMemorySourceFunction<>();
        Table modelDataTable =
                tEnv.fromDataStream(
                        env.addSource(
                                modelDataSource,
                                TypeInformation.of(LogisticRegressionModelDataSegment.class)));

        OnlineLogisticRegressionModel onlineModel =
                new OnlineLogisticRegressionModel()
                        .setModelData(modelDataTable)
                        .setFeaturesCol("features")
                        .setPredictionCol("prediction");
        transformAndOutputData(onlineModel, false);
        final JobID jobID = submitJob(env.getStreamGraph().getJobGraph());

        modelDataSource.addAll(modelData1);
        waitInitModelDataSetup(jobID);
        predictAndAssert(expectedRawInfo1, false);

        modelDataSource.addAll(modelData2);
        waitModelDataUpdate(jobID);
        predictAndAssert(expectedRawInfo2, false);
    }
}
