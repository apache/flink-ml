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

package org.apache.flink.ml.regression;

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.common.gbt.defs.TaskType;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressor;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressorModel;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
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
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static org.apache.flink.table.api.Expressions.$;

/** Tests {@link GBTRegressor} and {@link GBTRegressorModel}. */
public class GBTRegressorTest extends AbstractTestBase {
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
    List<Row> outputRows =
            Arrays.asList(
                    Row.of(40.06841194119824),
                    Row.of(40.94100994144195),
                    Row.of(40.93898887207972),
                    Row.of(40.14918141164082),
                    Row.of(40.90620397010659),
                    Row.of(40.06041865505043),
                    Row.of(40.1049148535624),
                    Row.of(40.88096567879293),
                    Row.of(40.08071914298763),
                    Row.of(40.86772065751431));

    private StreamTableEnvironment tEnv;
    private Table inputTable;

    private static void verifyPredictionResult(Table output, List<Row> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        //noinspection unchecked
        List<Row> results = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        final double delta = 1e-9;
        final Comparator<Row> comparator =
                Comparator.comparing(
                        d -> d.getFieldAs(0), new TestUtils.DoubleComparatorWithDelta(delta));
        TestUtils.compareResultCollectionsWithComparator(expected, results, comparator);
    }

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
        GBTRegressor gbtr = new GBTRegressor();
        Assert.assertArrayEquals(new String[] {"features"}, gbtr.getFeaturesCols());
        Assert.assertEquals("label", gbtr.getLabelCol());
        Assert.assertArrayEquals(new String[] {}, gbtr.getCategoricalCols());
        Assert.assertEquals("prediction", gbtr.getPredictionCol());

        Assert.assertNull(gbtr.getLeafCol());
        Assert.assertNull(gbtr.getWeightCol());
        Assert.assertEquals(5, gbtr.getMaxDepth());
        Assert.assertEquals(32, gbtr.getMaxBins());
        Assert.assertEquals(1, gbtr.getMinInstancesPerNode());
        Assert.assertEquals(0., gbtr.getMinWeightFractionPerNode(), 1e-12);
        Assert.assertEquals(0., gbtr.getMinInfoGain(), 1e-12);
        Assert.assertEquals(20, gbtr.getMaxIter());
        Assert.assertEquals(.1, gbtr.getStepSize(), 1e-12);
        Assert.assertEquals(GBTRegressor.class.getName().hashCode(), gbtr.getSeed());
        Assert.assertEquals(1., gbtr.getSubsamplingRate(), 1e-12);
        Assert.assertEquals("auto", gbtr.getFeatureSubsetStrategy());
        Assert.assertNull(gbtr.getValidationIndicatorCol());
        Assert.assertEquals(.01, gbtr.getValidationTol(), 1e-12);
        Assert.assertEquals(0., gbtr.getRegLambda(), 1e-12);
        Assert.assertEquals(1., gbtr.getRegGamma(), 1e-12);

        Assert.assertEquals("squared", gbtr.getLossType());

        gbtr.setFeaturesCols("f0", "f1", "f2")
                .setLabelCol("label")
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
                .setRegGamma(.1);

        Assert.assertArrayEquals(new String[] {"f0", "f1", "f2"}, gbtr.getFeaturesCols());
        Assert.assertEquals("label", gbtr.getLabelCol());
        Assert.assertArrayEquals(new String[] {"f0", "f1"}, gbtr.getCategoricalCols());
        Assert.assertEquals("pred", gbtr.getPredictionCol());

        Assert.assertEquals("leaf", gbtr.getLeafCol());
        Assert.assertEquals("weight", gbtr.getWeightCol());
        Assert.assertEquals(6, gbtr.getMaxDepth());
        Assert.assertEquals(64, gbtr.getMaxBins());
        Assert.assertEquals(2, gbtr.getMinInstancesPerNode());
        Assert.assertEquals(.1, gbtr.getMinWeightFractionPerNode(), 1e-12);
        Assert.assertEquals(.1, gbtr.getMinInfoGain(), 1e-12);
        Assert.assertEquals(10, gbtr.getMaxIter());
        Assert.assertEquals(.2, gbtr.getStepSize(), 1e-12);
        Assert.assertEquals(123, gbtr.getSeed());
        Assert.assertEquals(.8, gbtr.getSubsamplingRate(), 1e-12);
        Assert.assertEquals("0.5", gbtr.getFeatureSubsetStrategy());
        Assert.assertEquals("val", gbtr.getValidationIndicatorCol());
        Assert.assertEquals(.1, gbtr.getValidationTol(), 1e-12);
        Assert.assertEquals(.1, gbtr.getRegLambda(), 1e-12);
        Assert.assertEquals(.1, gbtr.getRegGamma(), 1e-12);
    }

    @Test
    public void testOutputSchema() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor().setFeaturesCols("f0", "f1", "f2").setCategoricalCols("f2");
        GBTRegressorModel model = gbtr.fit(inputTable);
        Table output = model.transform(inputTable)[0];
        Assert.assertArrayEquals(
                ArrayUtils.addAll(
                        inputTable.getResolvedSchema().getColumnNames().toArray(new String[0]),
                        gbtr.getPredictionCol()),
                output.getResolvedSchema().getColumnNames().toArray(new String[0]));
    }

    @Test
    public void testFitAndPredict() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTRegressorModel model = gbtr.fit(inputTable);
        Table output = model.transform(inputTable)[0].select($(gbtr.getPredictionCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testFitAndPredictWithVectorCol() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor()
                        .setFeaturesCols("vec")
                        .setLabelCol("label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTRegressorModel model = gbtr.fit(inputTable);
        Table output = model.transform(inputTable)[0].select($(gbtr.getPredictionCol()));
        List<Row> outputRowsUsingVectorCol =
                Arrays.asList(
                        Row.of(40.11011764668384),
                        Row.of(40.8838231947867),
                        Row.of(40.064839102170275),
                        Row.of(40.10374937485196),
                        Row.of(40.909914467915144),
                        Row.of(40.11472131282394),
                        Row.of(40.88106076252836),
                        Row.of(40.089859516616336),
                        Row.of(40.90833852360301),
                        Row.of(40.94920075468803));
        verifyPredictionResult(output, outputRowsUsingVectorCol);
    }

    @Test
    public void testFitAndPredictWithNoCategoricalCols() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor()
                        .setFeaturesCols("f0", "f1")
                        .setLabelCol("label")
                        .setRegGamma(0.)
                        .setMaxBins(5)
                        .setSeed(123);
        GBTRegressorModel model = gbtr.fit(inputTable);
        Table output = model.transform(inputTable)[0].select($(gbtr.getPredictionCol()));
        List<Row> outputRowsUsingNoCategoricalCols =
                Arrays.asList(
                        Row.of(40.07663214615239),
                        Row.of(40.92462268161843),
                        Row.of(40.941626445241624),
                        Row.of(40.06608854749729),
                        Row.of(40.12272436518743),
                        Row.of(40.92737873124178),
                        Row.of(40.08092204935494),
                        Row.of(40.898529570430696),
                        Row.of(40.08092204935494),
                        Row.of(40.88296818645738));
        verifyPredictionResult(output, outputRowsUsingNoCategoricalCols);
    }

    @Test
    public void testEstimatorSaveLoadAndPredict() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTRegressor loadedgbtr =
                TestUtils.saveAndReload(tEnv, gbtr, tempFolder.newFolder().getAbsolutePath());
        GBTRegressorModel model = loadedgbtr.fit(inputTable);
        Assert.assertEquals(
                Collections.singletonList("modelData"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Assert.assertEquals(
                Collections.singletonList("featureImportance"),
                model.getModelData()[1].getResolvedSchema().getColumnNames());
        Table output = model.transform(inputTable)[0].select($(gbtr.getPredictionCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testModelSaveLoadAndPredict() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTRegressorModel model = gbtr.fit(inputTable);
        GBTRegressorModel loadedModel =
                TestUtils.saveAndReload(tEnv, model, tempFolder.newFolder().getAbsolutePath());
        Table output = loadedModel.transform(inputTable)[0].select($(gbtr.getPredictionCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testGetModelData() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTRegressorModel model = gbtr.fit(inputTable);
        Table modelDataTable = model.getModelData()[0];
        List<String> modelDataColumnNames = modelDataTable.getResolvedSchema().getColumnNames();
        Assert.assertArrayEquals(
                new String[] {"modelData"}, modelDataColumnNames.toArray(new String[0]));

        //noinspection unchecked
        List<Row> modelDataRows =
                IteratorUtils.toList(tEnv.toDataStream(modelDataTable).executeAndCollect());
        Assert.assertEquals(1, modelDataRows.size());
        GBTModelData modelData = modelDataRows.get(0).getFieldAs(0);
        Assert.assertNotNull(modelData);

        Assert.assertEquals(TaskType.REGRESSION, TaskType.valueOf(modelData.type));
        Assert.assertFalse(modelData.isInputVector);
        Assert.assertEquals(40.5, modelData.prior, .5);
        Assert.assertEquals(gbtr.getStepSize(), modelData.stepSize, 1e-12);
        Assert.assertEquals(gbtr.getMaxIter(), modelData.allTrees.size());
        Assert.assertEquals(gbtr.getCategoricalCols().length, modelData.categoryToIdMaps.size());
        Assert.assertEquals(
                gbtr.getFeaturesCols().length - gbtr.getCategoricalCols().length,
                modelData.featureIdToBinEdges.size());
        Assert.assertEquals(BitSet.valueOf(new byte[] {4}), modelData.isCategorical);

        Table featureImportanceTable = model.getModelData()[1];
        Assert.assertEquals(
                Collections.singletonList("featureImportance"),
                featureImportanceTable.getResolvedSchema().getColumnNames());
        //noinspection unchecked
        List<Row> featureImportanceRows =
                IteratorUtils.toList(tEnv.toDataStream(featureImportanceTable).executeAndCollect());
        Assert.assertEquals(1, featureImportanceRows.size());
        Map<String, Double> featureImportanceMap =
                featureImportanceRows.get(0).getFieldAs("featureImportance");
        Assert.assertArrayEquals(
                gbtr.getFeaturesCols(), featureImportanceMap.keySet().toArray(new String[0]));
    }

    @Test
    public void testSetModelData() throws Exception {
        GBTRegressor gbtr =
                new GBTRegressor()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTRegressorModel modelA = gbtr.fit(inputTable);
        GBTRegressorModel modelB = new GBTRegressorModel().setModelData(modelA.getModelData());
        ReadWriteUtils.updateExistingParams(modelB, modelA.getParamMap());
        Table output = modelA.transform(inputTable)[0].select($(gbtr.getPredictionCol()));
        verifyPredictionResult(output, outputRows);
    }
}
