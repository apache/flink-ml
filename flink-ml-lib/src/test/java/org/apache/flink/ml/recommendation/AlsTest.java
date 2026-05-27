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

package org.apache.flink.ml.recommendation;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.recommendation.als.Als;
import org.apache.flink.ml.recommendation.als.AlsModel;
import org.apache.flink.ml.recommendation.als.AlsModelData;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link Als} and {@link AlsModel}. */
public class AlsTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;

    private StreamTableEnvironment tEnv;

    private final List<Row> trainData =
            Arrays.asList(
                    Row.of(1L, 5L, 0.1),
                    Row.of(2L, 8L, 0.5),
                    Row.of(3L, 5L, 0.8),
                    Row.of(4L, 7L, 0.1),
                    Row.of(1L, 7L, 0.7),
                    Row.of(2L, 5L, 0.9),
                    Row.of(3L, 8L, 0.1),
                    Row.of(2L, 6L, 0.7),
                    Row.of(2L, 7L, 0.4),
                    Row.of(1L, 8L, 0.3),
                    Row.of(4L, 6L, 0.4),
                    Row.of(3L, 7L, 0.6),
                    Row.of(1L, 6L, 0.5),
                    Row.of(4L, 8L, 0.3));

    private static final double TOLERANCE = 1.0e-7;
    private static final float FLOAT_TOLERANCE = 1.0e-6f;

    private final List<Row> smallTrainData =
            Arrays.asList(Row.of(1L, 5L, 0.7), Row.of(2L, 6L, 0.4));

    private final List<Row> testData = Collections.singletonList(Row.of(1L, 6L));

    private Table trainDataTable;
    private Table smallTrainDataTable;
    private Table testDataTable;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setParallelism(2);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);

        trainDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                trainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.LONG, Types.LONG, Types.DOUBLE
                                        },
                                        new String[] {"uid", "iid", "rating"})));

        smallTrainDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                smallTrainData,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.LONG, Types.LONG, Types.DOUBLE
                                        },
                                        new String[] {"uid", "iid", "rating"})));

        testDataTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                testData,
                                new RowTypeInfo(
                                        new TypeInformation[] {Types.LONG, Types.LONG},
                                        new String[] {"uid", "iid"})));
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(Table output, String predictionCol, double expectedData)
            throws Exception {
        List<Row> predictResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        for (Row predictionRow : predictResult) {
            double prediction = predictionRow.getFieldAs(predictionCol);
            assertEquals(prediction, expectedData, TOLERANCE);
        }
    }

    @Test
    public void testParam() {
        Als als = new Als();
        assertEquals("user", als.getUserCol());
        assertEquals("item", als.getItemCol());
        assertEquals("rating", als.getRatingCol());
        assertEquals(1.0, als.getAlpha(), TOLERANCE);
        assertEquals(0.1, als.getRegParam(), TOLERANCE);
        assertEquals(10, als.getRank());
        assertEquals(false, als.getImplicitPrefs());
        assertEquals(false, als.getNonNegative());
        assertEquals(10, als.getMaxIter());
        assertEquals("prediction", als.getPredictionCol());

        als.setUserCol("userCol")
                .setItemCol("itemCol")
                .setRatingCol("ratingCol")
                .setAlpha(0.001)
                .setRegParam(0.5)
                .setRank(100)
                .setImplicitPrefs(true)
                .setNonNegative(false)
                .setMaxIter(1000)
                .setPredictionCol("predict_result");

        assertEquals("userCol", als.getUserCol());
        assertEquals("itemCol", als.getItemCol());
        assertEquals("ratingCol", als.getRatingCol());
        assertEquals(0.001, als.getAlpha(), TOLERANCE);
        assertEquals(0.5, als.getRegParam(), TOLERANCE);
        assertEquals(100, als.getRank());
        assertEquals(true, als.getImplicitPrefs());
        assertEquals(false, als.getNonNegative());
        assertEquals(1000, als.getMaxIter());
        assertEquals("predict_result", als.getPredictionCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainDataTable.as("uid", "iid", "rating_col");
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating_col")
                        .setPredictionCol("predict_result");
        AlsModel model = als.fit(trainDataTable);
        Table output = model.transform(tempTable)[0];
        assertEquals(
                Arrays.asList("uid", "iid", "rating_col", "predict_result"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredictWithImplicit() throws Exception {
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating")
                        .setMaxIter(5)
                        .setRank(10)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setImplicitPrefs(true)
                        .setNonNegative(true)
                        .setPredictionCol("predict_result");
        Table output = als.fit(trainDataTable).transform(testDataTable)[0];
        verifyPredictionResult(output, als.getPredictionCol(), 0.8342121792822439);
    }

    @Test
    public void testFitAndPredictWithoutNonNegative() throws Exception {
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating")
                        .setMaxIter(5)
                        .setRank(10)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setImplicitPrefs(true)
                        .setNonNegative(false)
                        .setPredictionCol("predict_result");
        Table output = als.fit(trainDataTable).transform(testDataTable)[0];
        verifyPredictionResult(output, als.getPredictionCol(), 0.8345180300543498);
    }

    @Test
    public void testFitAndPredictWithExplicit() throws Exception {
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating")
                        .setMaxIter(5)
                        .setRank(10)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setImplicitPrefs(false)
                        .setNonNegative(true)
                        .setPredictionCol("predict_result");
        Table output = als.fit(trainDataTable).transform(testDataTable)[0];
        verifyPredictionResult(output, als.getPredictionCol(), 0.37476815535599206);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating")
                        .setMaxIter(10)
                        .setRank(10)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setImplicitPrefs(false)
                        .setNonNegative(true)
                        .setPredictionCol("predict_result");
        AlsModel model = als.fit(trainDataTable);
        AlsModel loadModel =
                TestUtils.saveAndReload(
                        tEnv, model, tempFolder.newFolder().getAbsolutePath(), AlsModel::load);
        Table output = loadModel.transform(testDataTable)[0];
        verifyPredictionResult(output, als.getPredictionCol(), 0.37558552399494904);
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testGetModelData() throws Exception {
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating")
                        .setMaxIter(10)
                        .setRank(3)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setImplicitPrefs(true)
                        .setNonNegative(true)
                        .setPredictionCol("predict_result");
        Table model = als.fit(trainDataTable).getModelData()[0];
        List<Row> modelRows = IteratorUtils.toList(tEnv.toDataStream(model).executeAndCollect());
        for (Row modelRow : modelRows) {
            AlsModelData modelData = modelRow.getFieldAs(0);
            for (Tuple2<Long, float[]> t2 : modelData.userFactors) {
                if (t2.f0 == 1L) {
                    assertArrayEquals(
                            t2.f1,
                            new float[] {0.72853327f, 0.33467698f, 0.59506977f},
                            FLOAT_TOLERANCE);
                }
                if (t2.f0 == 2L) {
                    assertArrayEquals(
                            t2.f1,
                            new float[] {0.7278192f, 0.33339077f, 0.60418415f},
                            FLOAT_TOLERANCE);
                }
                if (t2.f0 == 3L) {
                    assertArrayEquals(
                            t2.f1,
                            new float[] {0.15143539f, 0.82475346f, 0.5966393f},
                            FLOAT_TOLERANCE);
                }
                if (t2.f0 == 4L) {
                    assertArrayEquals(
                            t2.f1, new float[] {0.9454353f, 0.2567069f, 0.0f}, FLOAT_TOLERANCE);
                }
            }

            for (Tuple2<Long, float[]> t2 : modelData.itemFactors) {
                if (t2.f0 == 5L) {
                    assertArrayEquals(
                            t2.f1,
                            new float[] {0.16018498f, 0.42313296f, 0.9295262f},
                            FLOAT_TOLERANCE);
                }
                if (t2.f0 == 6L) {
                    assertArrayEquals(
                            t2.f1, new float[] {0.980295f, 0.0f, 0.19405676f}, FLOAT_TOLERANCE);
                }
                if (t2.f0 == 7L) {
                    assertArrayEquals(
                            t2.f1,
                            new float[] {0.7008933f, 0.5764357f, 0.41699994f},
                            FLOAT_TOLERANCE);
                }
                if (t2.f0 == 8L) {
                    assertArrayEquals(
                            t2.f1,
                            new float[] {0.7045122f, 0.57172996f, 0.41351604f},
                            FLOAT_TOLERANCE);
                }
            }
        }
    }

    @Test
    public void testSetModelData() throws Exception {
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating")
                        .setMaxIter(10)
                        .setRank(3)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setImplicitPrefs(true)
                        .setNonNegative(true)
                        .setPredictionCol("predict_result");
        Table modelData = als.fit(trainDataTable).getModelData()[0];

        AlsModel model =
                new AlsModel()
                        .setModelData(modelData)
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setPredictionCol("predict_result");

        Table output = model.transform(testDataTable)[0];
        verifyPredictionResult(output, als.getPredictionCol(), 0.8296548350206177);
    }

    @Test
    public void testMoreSubtaskThanData() throws Exception {
        env.setParallelism(4);
        Als als =
                new Als()
                        .setUserCol("uid")
                        .setItemCol("iid")
                        .setRatingCol("rating")
                        .setMaxIter(5)
                        .setRank(10)
                        .setAlpha(0.1)
                        .setRegParam(0.1)
                        .setImplicitPrefs(false)
                        .setNonNegative(true)
                        .setPredictionCol("predict_result");
        Table output = als.fit(smallTrainDataTable).transform(testDataTable)[0];
        verifyPredictionResult(output, als.getPredictionCol(), 0.3317218226859576);
    }
}
