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

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModel;
import org.apache.flink.ml.classification.logisticregression.LogisticRegressionModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.flink.ml.util.ReadWriteUtils.loadModelData;

/** Tests online model save and load. */
public class OnlineModelSaveLoadTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    StreamTableEnvironment tEnv;
    StreamExecutionEnvironment env;
    Schema modelSchema =
            Schema.newBuilder()
                    .column("f0", DataTypes.of(DenseVector.class))
                    .column("f1", DataTypes.of(long.class))
                    .column("f2", DataTypes.of(boolean.class))
                    .build();
    Schema dataSchema =
            Schema.newBuilder()
                    .column("f0", DataTypes.of(Double.class))
                    .column("f1", DataTypes.of(DenseVector.class))
                    .build();

    private static final List<Row> modelData =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(2.0, 4.5, 3.0), 1L, false),
                            Row.of(Vectors.dense(2.1, 4.6, 3.1), 1L, true),
                            Row.of(Vectors.dense(20.1, 5.6, 3.1), 3L, false),
                            Row.of(Vectors.dense(2.1, 4.7, 3.1), 3L, true)));

    private static final List<Row> validData =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(1.0, Vectors.dense(1.0, 3.5, -4.0)),
                            Row.of(0.0, Vectors.dense(1.1, -8.6, 3.3))));

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(1);
    }

    private String saveOnlineModel() throws Exception {
        String tmpPath = tempFolder.newFolder().getAbsolutePath();

        /* Constructs online LogisticRegression model. */
        Table lrModelStream =
                tEnv.fromDataStream(env.fromCollection(modelData), modelSchema)
                        .as("coefficient", "versionId", "isLastRecord");

        LogisticRegressionModel lr = new LogisticRegressionModel();
        /* Saves online model to given path (tmpPath). */
        lr.setModelData(lrModelStream);
        lr.save(tmpPath);
        env.execute();
        return tmpPath;
    }

    @Test
    public void saveAndLoadEverySingleModelAndValidate() throws Exception {
        String tmpPath = saveOnlineModel();
        /* Constructs validated data table. */
        Table validDataTable =
                tEnv.fromDataStream(env.fromCollection(validData), dataSchema).as("label, vec");

        Process proc = Runtime.getRuntime().exec("ls " + tmpPath + "/data");
        BufferedReader bufferedReader =
                new BufferedReader(new InputStreamReader(proc.getInputStream()));

        /* Loads every LogisticRegression model in model path and validates it. */
        String modelVersion;
        while ((modelVersion = bufferedReader.readLine()) != null) {
            System.out.println(modelVersion);
            LogisticRegressionModel lrModel = new LogisticRegressionModel().setFeaturesCol("vec");
            if (!"metadata".equals(modelVersion)) {
                Table modelData =
                        tEnv.fromDataStream(
                                        loadModelData(
                                                env,
                                                tmpPath,
                                                new LogisticRegressionModelData.ModelDataDecoder(),
                                                modelVersion))
                                .as("label, vec");
                lrModel.setModelData(modelData);
                Table out = lrModel.transform(validDataTable)[0];
                List<Row> result = IteratorUtils.toList(tEnv.toDataStream(out).executeAndCollect());
                for (Row row : result) {
                    Assert.assertEquals(row.getField(0), row.getField(2));
                }
            }
        }
    }

    @Test
    public void saveAndLoadAllOnlineModels() throws Exception {
        String tmpPath = saveOnlineModel();
        Table modelData =
                tEnv.fromDataStream(
                                loadModelData(
                                        env,
                                        tmpPath,
                                        new LogisticRegressionModelData.ModelDataDecoder()))
                        .as("label, vec");

        List<Row> result = IteratorUtils.toList(tEnv.toDataStream(modelData).executeAndCollect());
        Assert.assertEquals(result.size(), 4);
    }
}
