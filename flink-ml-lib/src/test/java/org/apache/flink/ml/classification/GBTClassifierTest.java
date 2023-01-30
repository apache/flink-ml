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
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifier;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifierModel;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.common.gbt.defs.TaskType;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.List;

/** Tests {@link GBTClassifier} and {@link GBTClassifierModel}. */
public class GBTClassifierTest extends AbstractTestBase {
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
    private StreamTableEnvironment tEnv;
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
        tEnv = StreamTableEnvironment.create(env);

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

    @Test
    public void testParam() {
        GBTClassifier gbtc = new GBTClassifier();
        Assert.assertEquals("features", gbtc.getFeaturesCol());
        Assert.assertNull(gbtc.getInputCols());
        Assert.assertEquals("label", gbtc.getLabelCol());
        Assert.assertArrayEquals(new String[] {}, gbtc.getCategoricalCols());
        Assert.assertEquals("prediction", gbtc.getPredictionCol());

        Assert.assertNull(gbtc.getLeafCol());
        Assert.assertNull(gbtc.getWeightCol());
        Assert.assertEquals(5, gbtc.getMaxDepth());
        Assert.assertEquals(32, gbtc.getMaxBins());
        Assert.assertEquals(1, gbtc.getMinInstancesPerNode());
        Assert.assertEquals(0., gbtc.getMinWeightFractionPerNode(), 1e-12);
        Assert.assertEquals(0., gbtc.getMinInfoGain(), 1e-12);
        Assert.assertEquals(20, gbtc.getMaxIter());
        Assert.assertEquals(.1, gbtc.getStepSize(), 1e-12);
        Assert.assertEquals(GBTClassifier.class.getName().hashCode(), gbtc.getSeed());
        Assert.assertEquals(1., gbtc.getSubsamplingRate(), 1e-12);
        Assert.assertEquals("auto", gbtc.getFeatureSubsetStrategy());
        Assert.assertNull(gbtc.getValidationIndicatorCol());
        Assert.assertEquals(.01, gbtc.getValidationTol(), 1e-12);
        Assert.assertEquals(0., gbtc.getRegLambda(), 1e-12);
        Assert.assertEquals(1., gbtc.getRegGamma(), 1e-12);

        Assert.assertEquals("logistic", gbtc.getLossType());
        Assert.assertEquals("rawPrediction", gbtc.getRawPredictionCol());
        Assert.assertEquals("probability", gbtc.getProbabilityCol());

        gbtc.setFeaturesCol("vec")
                .setInputCols("f0", "f1", "f2")
                .setLabelCol("cls_label")
                .setCategoricalCols("f0", "f1")
                .setPredictionCol("pred")
                .setLeafCol("leaf")
                .setWeightCol("weight")
                .setMaxDepth(6)
                .setMaxBins(64)
                .setMinInstancesPerNode(2)
                .setMinWeightFractionPerNode(.1)
                .setMinInfoGain(.1)
                .setMaxIter(10)
                .setStepSize(.2)
                .setSeed(123)
                .setSubsamplingRate(.8)
                .setFeatureSubsetStrategy("0.5")
                .setValidationIndicatorCol("val")
                .setValidationTol(.1)
                .setRegLambda(.1)
                .setRegGamma(.1)
                .setRawPredictionCol("raw_pred")
                .setProbabilityCol("prob");

        Assert.assertEquals("vec", gbtc.getFeaturesCol());
        Assert.assertArrayEquals(new String[] {"f0", "f1", "f2"}, gbtc.getInputCols());
        Assert.assertEquals("cls_label", gbtc.getLabelCol());
        Assert.assertArrayEquals(new String[] {"f0", "f1"}, gbtc.getCategoricalCols());
        Assert.assertEquals("pred", gbtc.getPredictionCol());

        Assert.assertEquals("leaf", gbtc.getLeafCol());
        Assert.assertEquals("weight", gbtc.getWeightCol());
        Assert.assertEquals(6, gbtc.getMaxDepth());
        Assert.assertEquals(64, gbtc.getMaxBins());
        Assert.assertEquals(2, gbtc.getMinInstancesPerNode());
        Assert.assertEquals(.1, gbtc.getMinWeightFractionPerNode(), 1e-12);
        Assert.assertEquals(.1, gbtc.getMinInfoGain(), 1e-12);
        Assert.assertEquals(10, gbtc.getMaxIter());
        Assert.assertEquals(.2, gbtc.getStepSize(), 1e-12);
        Assert.assertEquals(123, gbtc.getSeed());
        Assert.assertEquals(.8, gbtc.getSubsamplingRate(), 1e-12);
        Assert.assertEquals("0.5", gbtc.getFeatureSubsetStrategy());
        Assert.assertEquals("val", gbtc.getValidationIndicatorCol());
        Assert.assertEquals(.1, gbtc.getValidationTol(), 1e-12);
        Assert.assertEquals(.1, gbtc.getRegLambda(), 1e-12);
        Assert.assertEquals(.1, gbtc.getRegGamma(), 1e-12);

        Assert.assertEquals("raw_pred", gbtc.getRawPredictionCol());
        Assert.assertEquals("prob", gbtc.getProbabilityCol());
    }

    @Test
    public void testOutputSchema() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier().setInputCols("f0", "f1", "f2").setCategoricalCols("f2");
        GBTClassifierModel model = gbtc.fit(inputTable);
        Table output = model.transform(inputTable)[0];
        Assert.assertArrayEquals(
                ArrayUtils.addAll(
                        inputTable.getResolvedSchema().getColumnNames().toArray(new String[0]),
                        gbtc.getPredictionCol(),
                        gbtc.getRawPredictionCol(),
                        gbtc.getProbabilityCol()),
                output.getResolvedSchema().getColumnNames().toArray(new String[0]));
    }

    @Test
    public void testModelSaveLoadAndPredict() {
        // TODO: add test after complete model save/load methods in next PRs.
    }

    @Test
    public void testGetModelData() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setInputCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("cls_label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTClassifierModel model = gbtc.fit(inputTable);
        Table modelDataTable = model.getModelData()[0];
        List<String> modelDataColumnNames = modelDataTable.getResolvedSchema().getColumnNames();
        DataStream<Row> output = tEnv.toDataStream(modelDataTable);
        Assert.assertArrayEquals(
                new String[] {"modelData"}, modelDataColumnNames.toArray(new String[0]));

        Row modelDataRow = (Row) IteratorUtils.toList(output.executeAndCollect()).get(0);
        GBTModelData modelData = modelDataRow.getFieldAs(0);
        Assert.assertNotNull(modelData);

        Assert.assertEquals(TaskType.CLASSIFICATION, TaskType.valueOf(modelData.type));
        // TODO: check more fields in next PRs.
    }
}
