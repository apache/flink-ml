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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.TaskType;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.List;

/** Tests {@link GBTRunner}. */
public class GBTRunnerTest extends AbstractTestBase {
    private static final List<Row> inputDataRows =
            Arrays.asList(
                    Row.of(1.2, 2, null, 40., 1., 0., Vectors.dense(1.2, 2, Double.NaN)),
                    Row.of(2.3, 3, "b", 40., 2., 0., Vectors.dense(2.3, 3, 2.)),
                    Row.of(3.4, 4, "c", 40., 3., 0., Vectors.dense(3.4, 4, 3.)),
                    Row.of(4.5, 5, "a", 40., 4., 0., Vectors.dense(4.5, 5, 1.)),
                    Row.of(5.6, 2, "b", 40., 5., 0., Vectors.dense(5.6, 2, 2.)),
                    Row.of(null, 3, "c", 41., 1., 1., Vectors.dense(Double.NaN, 3, 3.)),
                    Row.of(12.8, 4, "e", 41., 2., 1., Vectors.dense(12.8, 4, 5.)),
                    Row.of(13.9, 2, "b", 41., 3., 1., Vectors.dense(13.9, 2, 2.)),
                    Row.of(14.1, 4, "a", 41., 4., 1., Vectors.dense(14.1, 4, 1.)),
                    Row.of(15.3, 1, "d", 41., 5., 1., Vectors.dense(15.3, 1, 4.)));

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private Table inputTable;

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.getConfig().enableObjectReuse();
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        inputTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                inputDataRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.DOUBLE,
                                            Types.INT,
                                            Types.STRING,
                                            Types.DOUBLE,
                                            Types.DOUBLE,
                                            Types.DOUBLE,
                                            VectorTypeInfo.INSTANCE
                                        },
                                        new String[] {
                                            "f0", "f1", "f2", "label", "weight", "cls_label", "vec"
                                        })));
    }

    private GbtParams getCommonGbtParams() {
        GbtParams p = new GbtParams();
        p.featureCols = new String[] {"f0", "f1", "f2"};
        p.categoricalCols = new String[] {"f2"};
        p.isInputVector = false;
        p.gamma = 0.;
        p.maxBins = 3;
        p.seed = 123;
        p.featureSubsetStrategy = "all";
        p.maxDepth = 3;
        p.maxNumLeaves = 1 << (p.maxDepth - 1);
        p.maxIter = 20;
        p.stepSize = 0.1;
        return p;
    }

    private void verifyModelData(GBTModelData modelData, GbtParams p) {
        Assert.assertEquals(p.taskType, TaskType.valueOf(modelData.type));
        Assert.assertEquals(p.stepSize, modelData.stepSize, 1e-12);
        Assert.assertEquals(p.maxIter, modelData.allTrees.size());
    }

    @Test
    public void testTrainClassifier() throws Exception {
        GbtParams p = getCommonGbtParams();
        p.taskType = TaskType.CLASSIFICATION;
        p.labelCol = "cls_label";
        p.lossType = "logistic";

        GBTModelData modelData = GBTRunner.train(inputTable, p).executeAndCollect().next();
        verifyModelData(modelData, p);
    }

    @Test
    public void testTrainRegressor() throws Exception {
        GbtParams p = getCommonGbtParams();
        p.taskType = TaskType.REGRESSION;
        p.labelCol = "label";
        p.lossType = "squared";

        GBTModelData modelData = GBTRunner.train(inputTable, p).executeAndCollect().next();
        verifyModelData(modelData, p);
    }
}
