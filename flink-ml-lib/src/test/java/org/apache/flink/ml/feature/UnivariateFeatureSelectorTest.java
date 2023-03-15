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

package org.apache.flink.ml.feature;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.feature.univariatefeatureselector.UnivariateFeatureSelector;
import org.apache.flink.ml.feature.univariatefeatureselector.UnivariateFeatureSelectorModel;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

/** Tests {@link UnivariateFeatureSelector} and {@link UnivariateFeatureSelectorModel}. */
public class UnivariateFeatureSelectorTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table inputChiSqTable;
    private Table inputANOVATable;
    private Table inputFValueTable;

    private static final double EPS = 1.0e-5;

    private UnivariateFeatureSelector selectorWithChiSqTest;
    private UnivariateFeatureSelector selectorWithANOVATest;
    private UnivariateFeatureSelector selectorWithFValueTest;

    private static final List<Row> INPUT_CHISQ_DATA =
            Arrays.asList(
                    Row.of(0.0, Vectors.dense(6.0, 7.0, 0.0, 7.0, 6.0, 0.0)),
                    Row.of(1.0, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0)),
                    Row.of(1.0, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0)),
                    Row.of(1.0, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0).toSparse()),
                    Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0).toSparse()),
                    Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 4.0, 0.0, 0.0).toSparse()));

    private static final List<Row> INPUT_ANOVA_DATA =
            Arrays.asList(
                    Row.of(
                            1,
                            Vectors.dense(
                                    4.65415496e-03,
                                    1.03550567e-01,
                                    -1.17358140e+00,
                                    1.61408773e-01,
                                    3.92492111e-01,
                                    7.31240882e-01)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    -9.01651741e-01,
                                    -5.28905302e-01,
                                    1.27636785e+00,
                                    7.02154563e-01,
                                    6.21348351e-01,
                                    1.88397353e-01)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    3.85692159e-01,
                                    -9.04639637e-01,
                                    5.09782604e-02,
                                    8.40043971e-01,
                                    7.45977857e-01,
                                    8.78402288e-01)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    1.36264353e+00,
                                    2.62454094e-01,
                                    7.96306202e-01,
                                    6.14948000e-01,
                                    7.44948187e-01,
                                    9.74034830e-01)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    9.65874070e-01,
                                    2.52773665e+00,
                                    -2.19380094e+00,
                                    2.33408080e-01,
                                    1.86340919e-01,
                                    8.23390433e-01)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    1.12324305e+01,
                                    -2.77121515e-01,
                                    1.12740513e-01,
                                    2.35184013e-01,
                                    3.46668895e-01,
                                    9.38500782e-02)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    1.06195839e+01,
                                    -1.82891238e+00,
                                    2.25085601e-01,
                                    9.09979851e-01,
                                    6.80257535e-02,
                                    8.24017480e-01)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    1.12806837e+01,
                                    1.30686889e+00,
                                    9.32839108e-02,
                                    3.49784755e-01,
                                    1.71322408e-02,
                                    7.48465194e-02)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    9.98689462e+00,
                                    9.50808938e-01,
                                    -2.90786359e-01,
                                    2.31253009e-01,
                                    7.46270968e-01,
                                    1.60308169e-01)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    1.08428551e+01,
                                    -1.02749936e+00,
                                    1.73951508e-01,
                                    8.92482744e-02,
                                    1.42651730e-01,
                                    7.66751625e-01)),
                    Row.of(
                            3,
                            Vectors.dense(
                                            -1.98641448e+00,
                                            1.12811990e+01,
                                            -2.35246756e-01,
                                            8.22809049e-01,
                                            3.26739456e-01,
                                            7.88268404e-01)
                                    .toSparse()),
                    Row.of(
                            3,
                            Vectors.dense(
                                            -6.09864090e-01,
                                            1.07346276e+01,
                                            -2.18805509e-01,
                                            7.33931213e-01,
                                            1.42554396e-01,
                                            7.11225605e-01)
                                    .toSparse()),
                    Row.of(
                            3,
                            Vectors.dense(
                                            -1.58481268e+00,
                                            9.19364039e+00,
                                            -5.87490459e-02,
                                            2.51532056e-01,
                                            2.82729807e-01,
                                            7.16245686e-01)
                                    .toSparse()),
                    Row.of(
                            3,
                            Vectors.dense(
                                            -2.50949277e-01,
                                            1.12815254e+01,
                                            -6.94806734e-01,
                                            5.93898886e-01,
                                            5.68425656e-01,
                                            8.49762330e-01)
                                    .toSparse()),
                    Row.of(
                            3,
                            Vectors.dense(
                                            7.63485129e-01,
                                            1.02605138e+01,
                                            1.32617719e+00,
                                            5.49682879e-01,
                                            8.59931442e-01,
                                            4.88677978e-02)
                                    .toSparse()),
                    Row.of(
                            4,
                            Vectors.dense(
                                            9.34900015e-01,
                                            4.11379043e-01,
                                            8.65010205e+00,
                                            9.23509168e-01,
                                            1.16995043e-01,
                                            5.91894106e-03)
                                    .toSparse()),
                    Row.of(
                            4,
                            Vectors.dense(
                                            4.73734933e-01,
                                            -1.48321181e+00,
                                            9.73349621e+00,
                                            4.09421563e-01,
                                            5.09375719e-01,
                                            5.93157850e-01)
                                    .toSparse()),
                    Row.of(
                            4,
                            Vectors.dense(
                                            3.41470679e-01,
                                            -6.88972582e-01,
                                            9.60347938e+00,
                                            3.62654055e-01,
                                            2.43437468e-01,
                                            7.13052838e-01)
                                    .toSparse()),
                    Row.of(
                            4,
                            Vectors.dense(
                                            -5.29614251e-01,
                                            -1.39262856e+00,
                                            1.01354144e+01,
                                            8.24123861e-01,
                                            5.84074506e-01,
                                            6.54461558e-01)
                                    .toSparse()),
                    Row.of(
                            4,
                            Vectors.dense(
                                            -2.99454508e-01,
                                            2.20457263e+00,
                                            1.14586015e+01,
                                            5.16336729e-01,
                                            9.99776159e-01,
                                            3.15769738e-01)
                                    .toSparse()));

    private static final List<Row> INPUT_FVALUE_DATA =
            Arrays.asList(
                    Row.of(
                            0.52516321,
                            Vectors.dense(
                                    0.19151945,
                                    0.62210877,
                                    0.43772774,
                                    0.78535858,
                                    0.77997581,
                                    0.27259261)),
                    Row.of(
                            0.88275782,
                            Vectors.dense(
                                    0.27646426,
                                    0.80187218,
                                    0.95813935,
                                    0.87593263,
                                    0.35781727,
                                    0.50099513)),
                    Row.of(
                            0.67524507,
                            Vectors.dense(
                                    0.68346294,
                                    0.71270203,
                                    0.37025075,
                                    0.56119619,
                                    0.50308317,
                                    0.01376845)),
                    Row.of(
                            0.76734745,
                            Vectors.dense(
                                    0.77282662,
                                    0.88264119,
                                    0.36488598,
                                    0.61539618,
                                    0.07538124,
                                    0.36882401)),
                    Row.of(
                            0.73909458,
                            Vectors.dense(
                                    0.9331401,
                                    0.65137814,
                                    0.39720258,
                                    0.78873014,
                                    0.31683612,
                                    0.56809865)),
                    Row.of(
                            0.83628141,
                            Vectors.dense(
                                    0.86912739,
                                    0.43617342,
                                    0.80214764,
                                    0.14376682,
                                    0.70426097,
                                    0.70458131)),
                    Row.of(
                            0.65665506,
                            Vectors.dense(
                                    0.21879211,
                                    0.92486763,
                                    0.44214076,
                                    0.90931596,
                                    0.05980922,
                                    0.18428708)),
                    Row.of(
                            0.58147135,
                            Vectors.dense(
                                    0.04735528,
                                    0.67488094,
                                    0.59462478,
                                    0.53331016,
                                    0.04332406,
                                    0.56143308)),
                    Row.of(
                            0.35603443,
                            Vectors.dense(
                                    0.32966845,
                                    0.50296683,
                                    0.11189432,
                                    0.60719371,
                                    0.56594464,
                                    0.00676406)),
                    Row.of(
                            0.94534373,
                            Vectors.dense(
                                    0.61744171,
                                    0.91212289,
                                    0.79052413,
                                    0.99208147,
                                    0.95880176,
                                    0.79196414)),
                    Row.of(
                            0.57458887,
                            Vectors.dense(
                                            0.28525096,
                                            0.62491671,
                                            0.4780938,
                                            0.19567518,
                                            0.38231745,
                                            0.05387369)
                                    .toSparse()),
                    Row.of(
                            0.59026777,
                            Vectors.dense(
                                            0.45164841,
                                            0.98200474,
                                            0.1239427,
                                            0.1193809,
                                            0.73852306,
                                            0.58730363)
                                    .toSparse()),
                    Row.of(
                            0.29894977,
                            Vectors.dense(
                                            0.47163253,
                                            0.10712682,
                                            0.22921857,
                                            0.89996519,
                                            0.41675354,
                                            0.53585166)
                                    .toSparse()),
                    Row.of(
                            0.34056582,
                            Vectors.dense(
                                            0.00620852,
                                            0.30064171,
                                            0.43689317,
                                            0.612149,
                                            0.91819808,
                                            0.62573667)
                                    .toSparse()),
                    Row.of(
                            0.64476446,
                            Vectors.dense(
                                            0.70599757,
                                            0.14983372,
                                            0.74606341,
                                            0.83100699,
                                            0.63372577,
                                            0.43830988)
                                    .toSparse()),
                    Row.of(
                            0.53724782,
                            Vectors.dense(
                                            0.15257277,
                                            0.56840962,
                                            0.52822428,
                                            0.95142876,
                                            0.48035918,
                                            0.50255956)
                                    .toSparse()),
                    Row.of(
                            0.5173021,
                            Vectors.dense(
                                            0.53687819,
                                            0.81920207,
                                            0.05711564,
                                            0.66942174,
                                            0.76711663,
                                            0.70811536)
                                    .toSparse()),
                    Row.of(
                            0.94508275,
                            Vectors.dense(
                                            0.79686718,
                                            0.55776083,
                                            0.96583653,
                                            0.1471569,
                                            0.029647,
                                            0.59389349)
                                    .toSparse()),
                    Row.of(
                            0.57739736,
                            Vectors.dense(
                                            0.1140657,
                                            0.95080985,
                                            0.96583653,
                                            0.19361869,
                                            0.45781165,
                                            0.92040257)
                                    .toSparse()),
                    Row.of(
                            0.53877145,
                            Vectors.dense(
                                            0.87906916,
                                            0.25261576,
                                            0.34800879,
                                            0.18258873,
                                            0.90179605,
                                            0.70652816)
                                    .toSparse()));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        selectorWithChiSqTest =
                new UnivariateFeatureSelector()
                        .setFeatureType("categorical")
                        .setLabelType("categorical");
        selectorWithANOVATest =
                new UnivariateFeatureSelector()
                        .setFeatureType("continuous")
                        .setLabelType("categorical");
        selectorWithFValueTest =
                new UnivariateFeatureSelector()
                        .setFeatureType("continuous")
                        .setLabelType("continuous");
        inputChiSqTable =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        INPUT_CHISQ_DATA,
                                        Types.ROW(Types.DOUBLE, VectorTypeInfo.INSTANCE)))
                        .as("label", "features");
        inputANOVATable =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        INPUT_ANOVA_DATA,
                                        Types.ROW(Types.INT, VectorTypeInfo.INSTANCE)))
                        .as("label", "features");
        inputFValueTable =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        INPUT_FVALUE_DATA,
                                        Types.ROW(Types.DOUBLE, VectorTypeInfo.INSTANCE)))
                        .as("label", "features");
    }

    private void transformAndVerify(
            UnivariateFeatureSelector selector, Table table, int... expectedIndices)
            throws Exception {
        UnivariateFeatureSelectorModel model = selector.fit(table);
        Table output = model.transform(table)[0];
        verifyOutputResult(output, expectedIndices);
    }

    private void verifyOutputResult(Table table, int... expectedIndices) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) table).getTableEnvironment();
        CloseableIterator<Row> rowIterator = tEnv.toDataStream(table).executeAndCollect();
        while (rowIterator.hasNext()) {
            Row row = rowIterator.next();
            assertEquals(expectedIndices.length, ((Vector) row.getField("output")).size());
            for (int i = 0; i < expectedIndices.length; i++) {
                assertEquals(
                        ((Vector) row.getField("features")).get(expectedIndices[i]),
                        ((Vector) row.getField("output")).get(i),
                        EPS);
            }
        }
    }

    @Test
    public void testParam() {
        UnivariateFeatureSelector selector = new UnivariateFeatureSelector();
        assertEquals("features", selector.getFeaturesCol());
        assertEquals("label", selector.getLabelCol());
        assertEquals("output", selector.getOutputCol());
        assertEquals("numTopFeatures", selector.getSelectionMode());
        assertNull(selector.getSelectionThreshold());

        selector.setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setOutputCol("test_output")
                .setFeatureType("continuous")
                .setLabelType("categorical")
                .setSelectionMode("fpr")
                .setSelectionThreshold(0.01);

        assertEquals("test_features", selector.getFeaturesCol());
        assertEquals("test_label", selector.getLabelCol());
        assertEquals("test_output", selector.getOutputCol());
        assertEquals("continuous", selector.getFeatureType());
        assertEquals("categorical", selector.getLabelType());
        assertEquals("fpr", selector.getSelectionMode());
        assertEquals(0.01, selector.getSelectionThreshold(), EPS);
    }

    @Test
    public void testIncompatibleSelectionModeAndThreshold() {
        UnivariateFeatureSelector selector =
                new UnivariateFeatureSelector()
                        .setFeatureType("continuous")
                        .setLabelType("categorical")
                        .setSelectionThreshold(50.1);

        try {
            selector.fit(inputANOVATable);
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "SelectionThreshold needs to be a positive Integer "
                            + "for selection mode numTopFeatures, but got 50.1.",
                    e.getMessage());
        }
        try {
            selector.setSelectionMode("fpr").setSelectionThreshold(1.1).fit(inputANOVATable);
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "SelectionThreshold needs to be in the range [0, 1] "
                            + "for selection mode fpr, but got 1.1.",
                    e.getMessage());
        }
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = inputANOVATable.as("test_label", "test_features");
        UnivariateFeatureSelector selector =
                new UnivariateFeatureSelector()
                        .setLabelCol("test_label")
                        .setFeaturesCol("test_features")
                        .setOutputCol("test_output")
                        .setFeatureType("continuous")
                        .setLabelType("categorical");

        UnivariateFeatureSelectorModel model = selector.fit(tempTable);
        Table output = model.transform(tempTable)[0];
        assertEquals(
                Arrays.asList("test_label", "test_features", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitTransformWithNumTopFeatures() throws Exception {
        transformAndVerify(selectorWithChiSqTest.setSelectionThreshold(2), inputChiSqTable, 0, 1);
        transformAndVerify(selectorWithANOVATest.setSelectionThreshold(2), inputANOVATable, 0, 2);
        transformAndVerify(selectorWithFValueTest.setSelectionThreshold(2), inputFValueTable, 0, 2);
    }

    @Test
    public void testFitTransformWithPercentile() throws Exception {
        transformAndVerify(
                selectorWithChiSqTest.setSelectionMode("percentile").setSelectionThreshold(0.17),
                inputChiSqTable,
                0);
        transformAndVerify(
                selectorWithANOVATest.setSelectionMode("percentile").setSelectionThreshold(0.17),
                inputANOVATable,
                0);
        transformAndVerify(
                selectorWithFValueTest.setSelectionMode("percentile").setSelectionThreshold(0.17),
                inputFValueTable,
                2);
    }

    @Test
    public void testFitTransformWithFPR() throws Exception {
        transformAndVerify(
                selectorWithChiSqTest.setSelectionMode("fpr").setSelectionThreshold(0.02),
                inputChiSqTable,
                0);
        transformAndVerify(
                selectorWithANOVATest.setSelectionMode("fpr").setSelectionThreshold(1.0E-12),
                inputANOVATable,
                0);
        transformAndVerify(
                selectorWithFValueTest.setSelectionMode("fpr").setSelectionThreshold(0.01),
                inputFValueTable,
                2);
    }

    @Test
    public void testFitTransformWithFDR() throws Exception {
        transformAndVerify(
                selectorWithChiSqTest.setSelectionMode("fdr").setSelectionThreshold(0.12),
                inputChiSqTable,
                0);
        transformAndVerify(
                selectorWithANOVATest.setSelectionMode("fdr").setSelectionThreshold(6.0E-12),
                inputANOVATable,
                0);
        transformAndVerify(
                selectorWithFValueTest.setSelectionMode("fdr").setSelectionThreshold(0.03),
                inputFValueTable,
                2);
    }

    @Test
    public void testFitTransformWithFWE() throws Exception {
        transformAndVerify(
                selectorWithChiSqTest.setSelectionMode("fwe").setSelectionThreshold(0.12),
                inputChiSqTable,
                0);
        transformAndVerify(
                selectorWithANOVATest.setSelectionMode("fwe").setSelectionThreshold(6.0E-12),
                inputANOVATable,
                0);
        transformAndVerify(
                selectorWithFValueTest.setSelectionMode("fwe").setSelectionThreshold(0.03),
                inputFValueTable,
                2);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        UnivariateFeatureSelector selector =
                new UnivariateFeatureSelector()
                        .setFeatureType("continuous")
                        .setLabelType("categorical")
                        .setSelectionThreshold(1);

        UnivariateFeatureSelector loadSelector =
                TestUtils.saveAndReload(
                        tEnv,
                        selector,
                        tempFolder.newFolder().getAbsolutePath(),
                        UnivariateFeatureSelector::load);
        UnivariateFeatureSelectorModel model = loadSelector.fit(inputANOVATable);
        UnivariateFeatureSelectorModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        UnivariateFeatureSelectorModel::load);
        assertEquals(
                Collections.singletonList("indices"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        Table output = loadedModel.transform(inputANOVATable)[0];
        verifyOutputResult(output, 0);
    }

    @Test
    public void testIncompatibleNumOfFeatures() {
        UnivariateFeatureSelector selector =
                new UnivariateFeatureSelector()
                        .setFeatureType("continuous")
                        .setLabelType("continuous")
                        .setSelectionThreshold(1);
        UnivariateFeatureSelectorModel model = selector.fit(inputFValueTable);

        List<Row> predictData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(1, Vectors.dense(1.0, 2.0)),
                                Row.of(-1, Vectors.dense(-1.0, -2.0))));
        Table predictTable =
                tEnv.fromDataStream(env.fromCollection(predictData)).as("label", "features");
        Table output = model.transform(predictTable)[0];
        try {
            output.execute().print();
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "Input 2 features, but UnivariateFeatureSelector is "
                            + "expecting at least 3 features as input.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testEquivalentPValues() throws Exception {
        List<Row> inputData =
                Arrays.asList(
                        Row.of(0.0, Vectors.dense(6.0, 7.0, 0.0, 6.0, 6.0, 6.0)),
                        Row.of(1.0, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 0.0)),
                        Row.of(1.0, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 0.0)),
                        Row.of(1.0, Vectors.dense(0.0, 9.0, 8.0, 0.0, 6.0, 0.0)),
                        Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 8.0, 4.0, 8.0)),
                        Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 8.0, 0.0, 8.0)));
        Table inputTable =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        inputData,
                                        Types.ROW(Types.DOUBLE, VectorTypeInfo.INSTANCE)))
                        .as("label", "features");
        UnivariateFeatureSelectorModel model =
                selectorWithChiSqTest.setSelectionThreshold(4).fit(inputTable);
        Table modelData = model.getModelData()[0];
        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        int[] expectedIndices = {0, 3, 5, 1};
        assertArrayEquals(expectedIndices, (int[]) modelRows.get(0).getField(0));
    }

    @Test
    public void testGetModelData() throws Exception {
        UnivariateFeatureSelector selector =
                new UnivariateFeatureSelector()
                        .setFeatureType("continuous")
                        .setLabelType("categorical")
                        .setSelectionThreshold(3);
        UnivariateFeatureSelectorModel model = selector.fit(inputANOVATable);
        Table modelData = model.getModelData()[0];
        assertEquals(
                Collections.singletonList("indices"),
                modelData.getResolvedSchema().getColumnNames());

        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        int[] expectedIndices = {0, 2, 1};
        assertArrayEquals(expectedIndices, (int[]) modelRows.get(0).getField(0));
    }

    @Test
    public void testSetModelData() throws Exception {
        UnivariateFeatureSelector selector =
                selectorWithANOVATest.setSelectionMode("fpr").setSelectionThreshold(1.0E-12);
        UnivariateFeatureSelectorModel modelA = selector.fit(inputANOVATable);
        Table modelData = modelA.getModelData()[0];

        UnivariateFeatureSelectorModel modelB =
                new UnivariateFeatureSelectorModel().setModelData(modelData);
        Table output = modelB.transform(inputANOVATable)[0];
        verifyOutputResult(output, 0);
    }
}
