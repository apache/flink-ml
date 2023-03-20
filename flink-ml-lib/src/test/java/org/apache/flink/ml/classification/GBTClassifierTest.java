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
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
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

import static org.apache.flink.table.api.Expressions.$;

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
    List<Row> outputRows =
            Arrays.asList(
                    Row.of(
                            0.0,
                            Vectors.dense(2.376078066514637, -2.376078066514637),
                            Vectors.dense(0.914984852695779, 0.08501514730422102)),
                    Row.of(
                            1.0,
                            Vectors.dense(-2.5493892913102703, 2.5493892913102703),
                            Vectors.dense(0.07246752402942669, 0.9275324759705733)),
                    Row.of(
                            1.0,
                            Vectors.dense(-2.658830586839206, 2.658830586839206),
                            Vectors.dense(0.06544682253255263, 0.9345531774674474)),
                    Row.of(
                            0.0,
                            Vectors.dense(2.3309355512336296, -2.3309355512336296),
                            Vectors.dense(0.9114069063091061, 0.08859309369089385)),
                    Row.of(
                            1.0,
                            Vectors.dense(-2.6577392865785714, 2.6577392865785714),
                            Vectors.dense(0.06551360197733425, 0.9344863980226658)),
                    Row.of(
                            0.0,
                            Vectors.dense(2.5532653631402114, -2.5532653631402114),
                            Vectors.dense(0.9277925785910718, 0.07220742140892823)),
                    Row.of(
                            0.0,
                            Vectors.dense(2.3773197509703996, -2.3773197509703996),
                            Vectors.dense(0.9150813905583675, 0.0849186094416325)),
                    Row.of(
                            1.0,
                            Vectors.dense(-2.132645378098387, 2.132645378098387),
                            Vectors.dense(0.10596411850817689, 0.8940358814918231)),
                    Row.of(
                            0.0,
                            Vectors.dense(2.3105035625447106, -2.3105035625447106),
                            Vectors.dense(0.9097432116019103, 0.09025678839808973)),
                    Row.of(
                            1.0,
                            Vectors.dense(-2.0541952729346695, 2.0541952729346695),
                            Vectors.dense(0.11362915817869357, 0.8863708418213064)));
    private StreamTableEnvironment tEnv;
    private Table inputTable;

    private static void verifyPredictionResult(Table output, List<Row> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        //noinspection unchecked
        List<Row> results = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        final double delta = 1e-3;
        final Comparator<DenseVector> denseVectorComparator =
                new TestUtils.DenseVectorComparatorWithDelta(delta);
        final Comparator<Row> comparator =
                Comparator.<Row, Double>comparing(d -> d.getFieldAs(0))
                        .thenComparing(d -> d.getFieldAs(1), denseVectorComparator)
                        .thenComparing(d -> d.getFieldAs(2), denseVectorComparator);
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
        GBTClassifier gbtc = new GBTClassifier();
        Assert.assertArrayEquals(new String[] {"features"}, gbtc.getFeaturesCols());
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

        gbtc.setFeaturesCols("f0", "f1", "f2")
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

        Assert.assertArrayEquals(new String[] {"f0", "f1", "f2"}, gbtc.getFeaturesCols());
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
                new GBTClassifier().setFeaturesCols("f0", "f1", "f2").setCategoricalCols("f2");
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
    public void testFitAndPredict() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("cls_label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTClassifierModel model = gbtc.fit(inputTable);
        Table output =
                model.transform(inputTable)[0].select(
                        $(gbtc.getPredictionCol()),
                        $(gbtc.getRawPredictionCol()),
                        $(gbtc.getProbabilityCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testFitAndPredictWithVectorCol() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setFeaturesCols("vec")
                        .setLabelCol("cls_label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTClassifierModel model = gbtc.fit(inputTable);
        Table output =
                model.transform(inputTable)[0].select(
                        $(gbtc.getPredictionCol()),
                        $(gbtc.getRawPredictionCol()),
                        $(gbtc.getProbabilityCol()));
        List<Row> outputRowsUsingVectorCol =
                Arrays.asList(
                        Row.of(
                                0.0,
                                Vectors.dense(1.9834935486026828, -1.9834935486026828),
                                Vectors.dense(0.8790530839977041, 0.12094691600229594)),
                        Row.of(
                                1.0,
                                Vectors.dense(-1.9962334686995544, 1.9962334686995544),
                                Vectors.dense(0.11959895119804398, 0.880401048801956)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.2596958412285053, -2.2596958412285053),
                                Vectors.dense(0.9054836034255209, 0.0945163965744791)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.23023965816558, 2.23023965816558),
                                Vectors.dense(0.09706763399626683, 0.9029323660037332)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.520667396406638, 2.520667396406638),
                                Vectors.dense(0.0744219596185437, 0.9255780403814563)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.5005544570205114, -2.5005544570205114),
                                Vectors.dense(0.9241806803368346, 0.07581931966316532)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.155310746068554, -2.155310746068554),
                                Vectors.dense(0.8961640042377698, 0.10383599576223027)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.2386996519306424, 2.2386996519306424),
                                Vectors.dense(0.09632867690962832, 0.9036713230903717)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.0375281995821273, -2.0375281995821273),
                                Vectors.dense(0.8846813338862343, 0.11531866611376576)),
                        Row.of(
                                1.0,
                                Vectors.dense(-1.9751553623558855, 1.9751553623558855),
                                Vectors.dense(0.12183622723878906, 0.8781637727612109)));
        verifyPredictionResult(output, outputRowsUsingVectorCol);
    }

    @Test
    public void testFitAndPredictWithNoCategoricalCols() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setFeaturesCols("f0", "f1")
                        .setLabelCol("cls_label")
                        .setRegGamma(0.)
                        .setMaxBins(5)
                        .setSeed(123);
        GBTClassifierModel model = gbtc.fit(inputTable);
        Table output =
                model.transform(inputTable)[0].select(
                        $(gbtc.getPredictionCol()),
                        $(gbtc.getRawPredictionCol()),
                        $(gbtc.getProbabilityCol()));
        List<Row> outputRowsUsingNoCategoricalCols =
                Arrays.asList(
                        Row.of(
                                0.0,
                                Vectors.dense(2.4386858360079877, -2.4386858360079877),
                                Vectors.dense(0.9197301210345855, 0.08026987896541447)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.079593609142336, -2.079593609142336),
                                Vectors.dense(0.8889039070093702, 0.11109609299062985)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.4477766607449594, 2.4477766607449594),
                                Vectors.dense(0.07960128978764613, 0.9203987102123539)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.3680506847981113, -2.3680506847981113),
                                Vectors.dense(0.9143583384561507, 0.0856416615438493)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.0115161495245792, 2.0115161495245792),
                                Vectors.dense(0.11799909267017583, 0.8820009073298242)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.3680506847981113, -2.3680506847981113),
                                Vectors.dense(0.9143583384561507, 0.0856416615438493)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.1774376078697983, 2.1774376078697983),
                                Vectors.dense(0.10179497553813543, 0.8982050244618646)),
                        Row.of(
                                0.0,
                                Vectors.dense(2.434832949283468, -2.434832949283468),
                                Vectors.dense(0.9194452150195366, 0.08055478498046341)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.441225164856452, 2.441225164856452),
                                Vectors.dense(0.08008260858505134, 0.9199173914149487)),
                        Row.of(
                                1.0,
                                Vectors.dense(-2.672457199454413, 2.672457199454413),
                                Vectors.dense(0.06461828968951666, 0.9353817103104833)));
        verifyPredictionResult(output, outputRowsUsingNoCategoricalCols);
    }

    @Test
    public void testEstimatorSaveLoadAndPredict() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("cls_label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTClassifier loadedGbtc =
                TestUtils.saveAndReload(tEnv, gbtc, tempFolder.newFolder().getAbsolutePath());
        GBTClassifierModel model = loadedGbtc.fit(inputTable);
        Assert.assertEquals(
                Collections.singletonList("modelData"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output =
                model.transform(inputTable)[0].select(
                        $(gbtc.getPredictionCol()),
                        $(gbtc.getRawPredictionCol()),
                        $(gbtc.getProbabilityCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testModelSaveLoadAndPredict() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("cls_label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTClassifierModel model = gbtc.fit(inputTable);
        GBTClassifierModel loadedModel =
                TestUtils.saveAndReload(tEnv, model, tempFolder.newFolder().getAbsolutePath());
        Table output =
                loadedModel.transform(inputTable)[0].select(
                        $(gbtc.getPredictionCol()),
                        $(gbtc.getRawPredictionCol()),
                        $(gbtc.getProbabilityCol()));
        verifyPredictionResult(output, outputRows);
    }

    @Test
    public void testGetModelData() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setFeaturesCols("f0", "f1", "f2")
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
        Assert.assertFalse(modelData.isInputVector);
        Assert.assertEquals(0., modelData.prior, 1e-12);
        Assert.assertEquals(gbtc.getStepSize(), modelData.stepSize, 1e-12);
        Assert.assertEquals(gbtc.getMaxIter(), modelData.allTrees.size());
        Assert.assertEquals(gbtc.getCategoricalCols().length, modelData.categoryToIdMaps.size());
        Assert.assertEquals(
                gbtc.getFeaturesCols().length - gbtc.getCategoricalCols().length,
                modelData.featureIdToBinEdges.size());
        Assert.assertEquals(BitSet.valueOf(new byte[] {4}), modelData.isCategorical);
    }

    @Test
    public void testSetModelData() throws Exception {
        GBTClassifier gbtc =
                new GBTClassifier()
                        .setFeaturesCols("f0", "f1", "f2")
                        .setCategoricalCols("f2")
                        .setLabelCol("cls_label")
                        .setRegGamma(0.)
                        .setMaxBins(3)
                        .setSeed(123);
        GBTClassifierModel modelA = gbtc.fit(inputTable);
        GBTClassifierModel modelB = new GBTClassifierModel().setModelData(modelA.getModelData());
        ReadWriteUtils.updateExistingParams(modelB, modelA.getParamMap());
        Table output =
                modelA.transform(inputTable)[0].select(
                        $(gbtc.getPredictionCol()),
                        $(gbtc.getRawPredictionCol()),
                        $(gbtc.getProbabilityCol()));
        verifyPredictionResult(output, outputRows);
    }
}
