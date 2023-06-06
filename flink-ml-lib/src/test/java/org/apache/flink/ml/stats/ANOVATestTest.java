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

package org.apache.flink.ml.stats;

import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.stats.anovatest.ANOVATest;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Tests the {@link ANOVATest}. */
public class ANOVATestTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private Table denseInputTable;
    private Table sparseInputTable;

    private static final double EPS = 1.0e-5;
    private static final List<Row> DENSE_INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            3,
                            Vectors.dense(
                                    0.85956061,
                                    0.1645695,
                                    0.48347596,
                                    0.92102727,
                                    0.42855644,
                                    0.05746009)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    0.92500743,
                                    0.65760154,
                                    0.13295284,
                                    0.53344893,
                                    0.8994776,
                                    0.24836496)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    0.03017182,
                                    0.07244715,
                                    0.87416449,
                                    0.55843035,
                                    0.91604736,
                                    0.63346045)),
                    Row.of(
                            5,
                            Vectors.dense(
                                    0.28325261,
                                    0.36536881,
                                    0.09223386,
                                    0.37251258,
                                    0.34742278,
                                    0.70517077)),
                    Row.of(
                            4,
                            Vectors.dense(
                                    0.64850904,
                                    0.04090877,
                                    0.21173176,
                                    0.00148992,
                                    0.13897166,
                                    0.21182539)),
                    Row.of(
                            4,
                            Vectors.dense(
                                    0.02609493,
                                    0.44608735,
                                    0.23910531,
                                    0.95449222,
                                    0.90763182,
                                    0.8624905)),
                    Row.of(
                            5,
                            Vectors.dense(
                                    0.09158744,
                                    0.97745235,
                                    0.41150139,
                                    0.45830467,
                                    0.52590925,
                                    0.29441554)),
                    Row.of(
                            4,
                            Vectors.dense(
                                    0.97211594,
                                    0.1814442,
                                    0.30340642,
                                    0.17445413,
                                    0.52756958,
                                    0.02069296)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    0.06354593,
                                    0.63527231,
                                    0.49620335,
                                    0.0141264,
                                    0.62722219,
                                    0.63497507)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    0.10814149,
                                    0.8296426,
                                    0.51775217,
                                    0.57068344,
                                    0.54633305,
                                    0.12714921)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    0.72731796,
                                    0.94010124,
                                    0.45007811,
                                    0.87650674,
                                    0.53735565,
                                    0.49568415)),
                    Row.of(
                            2,
                            Vectors.dense(
                                    0.41827208,
                                    0.85100628,
                                    0.38685271,
                                    0.60689503,
                                    0.21784097,
                                    0.91294433)),
                    Row.of(
                            3,
                            Vectors.dense(
                                    0.65843656,
                                    0.5880859,
                                    0.18862706,
                                    0.856398,
                                    0.18029327,
                                    0.94851926)),
                    Row.of(
                            4,
                            Vectors.dense(
                                    0.3841634,
                                    0.25138793,
                                    0.96746644,
                                    0.77048045,
                                    0.44685196,
                                    0.19813854)),
                    Row.of(
                            5,
                            Vectors.dense(
                                    0.65982267,
                                    0.23024125,
                                    0.13598434,
                                    0.60144265,
                                    0.57848927,
                                    0.85623564)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    0.35764189,
                                    0.47623815,
                                    0.5459232,
                                    0.79508298,
                                    0.14462443,
                                    0.01802919)),
                    Row.of(
                            5,
                            Vectors.dense(
                                    0.38532153,
                                    0.90614554,
                                    0.86629571,
                                    0.13988735,
                                    0.32062385,
                                    0.00179492)),
                    Row.of(
                            3,
                            Vectors.dense(
                                    0.2142368,
                                    0.28306022,
                                    0.59481646,
                                    0.42567028,
                                    0.52207663,
                                    0.78082401)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    0.20788283,
                                    0.76861782,
                                    0.59595468,
                                    0.62103642,
                                    0.17781246,
                                    0.77655345)),
                    Row.of(
                            1,
                            Vectors.dense(
                                    0.1751708,
                                    0.4547537,
                                    0.46187865,
                                    0.79781199,
                                    0.05104487,
                                    0.42406092)));

    private static final List<Row> SPARSE_INPUT_DATA =
            Arrays.asList(
                    Row.of(3, Vectors.dense(6.0, 7.0, 0.0, 7.0, 6.0, 0.0, 0.0).toSparse()),
                    Row.of(1, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0, 0.0).toSparse()),
                    Row.of(3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0, 0.0).toSparse()),
                    Row.of(2, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0, 0.0).toSparse()),
                    Row.of(2, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0, 0.0).toSparse()),
                    Row.of(3, Vectors.dense(Double.NaN, 9.0, 6.0, 4.0, 0.0, 0.0, 0.0).toSparse()));

    private static final Row EXPECTED_OUTPUT_DENSE =
            Row.of(
                    Vectors.dense(
                            0.64137831, 0.14830724, 0.69858474, 0.28038169, 0.86759161, 0.81608606),
                    new long[] {19, 19, 19, 19, 19, 19},
                    Vectors.dense(
                            0.64110932, 1.98689258, 0.55499714, 1.40340562, 0.30881722, 0.3848595));

    private static final List<Row> EXPECTED_FLATTENED_OUTPUT_DENSE =
            Arrays.asList(
                    Row.of(0, 0.64137831, 19, 0.64110932),
                    Row.of(1, 0.14830724, 19, 1.98689258),
                    Row.of(2, 0.69858474, 19, 0.55499714),
                    Row.of(3, 0.28038169, 19, 1.40340562),
                    Row.of(4, 0.86759161, 19, 0.30881722),
                    Row.of(5, 0.81608606, 19, 0.3848595));

    private static final Row EXPECTED_OUTPUT_SPARSE =
            Row.of(
                    Vectors.dense(
                            Double.NaN,
                            0.71554175,
                            0.34278574,
                            0.45824059,
                            0.84633632,
                            0.15673368,
                            Double.NaN),
                    new long[] {5, 5, 5, 5, 5, 5, 5},
                    Vectors.dense(
                            Double.NaN, 0.375, 1.5625, 1.02364865, 0.17647059, 3.66, Double.NaN));

    private static final List<Row> EXPECTED_FLATTENED_OUTPUT_SPARSE =
            Arrays.asList(
                    Row.of(0, Double.NaN, 5, Double.NaN),
                    Row.of(1, 0.71554175, 5, 0.375),
                    Row.of(2, 0.34278574, 5, 1.5625),
                    Row.of(3, 0.45824059, 5, 1.02364865),
                    Row.of(4, 0.84633632, 5, 0.17647059),
                    Row.of(5, 0.15673368, 5, 3.66),
                    Row.of(6, Double.NaN, 5, Double.NaN));

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        denseInputTable =
                tEnv.fromDataStream(env.fromCollection(DENSE_INPUT_DATA)).as("label", "features");
        sparseInputTable =
                tEnv.fromDataStream(env.fromCollection(SPARSE_INPUT_DATA)).as("label", "features");
    }

    private static void verifyFlattenTransformationResult(Table output, List<Row> expected)
            throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Row> outputDataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(outputDataStream.executeAndCollect());
        assertEquals(expected.size(), results.size());

        results.sort(Comparator.comparing(r -> String.valueOf(r.getField(0))));
        expected.sort(Comparator.comparing(r -> String.valueOf(r.getField(0))));

        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i).getArity(), results.get(i).getArity());
            for (int j = 0; j < expected.get(i).getArity(); j++) {
                assertEquals(
                        Double.valueOf(expected.get(i).getField(j).toString()),
                        Double.valueOf(results.get(i).getField(j).toString()),
                        EPS);
            }
        }
    }

    private static void verifyTransformationResult(Table output, Row expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Row> outputDataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(outputDataStream.executeAndCollect());
        assertEquals(1, results.size());

        Row result = results.get(0);
        assertEquals(3, result.getArity());
        assertArrayEquals(
                ((IntDoubleVector) expected.getField(0)).toArray(),
                ((IntDoubleVector) result.getField(0)).toArray(),
                EPS);
        assertArrayEquals((long[]) expected.getField(1), (long[]) result.getField(1));
        assertArrayEquals(
                ((IntDoubleVector) expected.getField(2)).toArray(),
                ((IntDoubleVector) result.getField(2)).toArray(),
                EPS);
    }

    @Test
    public void testParam() {
        ANOVATest anovaTest = new ANOVATest();
        assertEquals("label", anovaTest.getLabelCol());
        assertEquals("features", anovaTest.getFeaturesCol());
        assertFalse(anovaTest.getFlatten());

        anovaTest.setLabelCol("test_label").setFeaturesCol("test_features").setFlatten(true);

        assertEquals("test_features", anovaTest.getFeaturesCol());
        assertEquals("test_label", anovaTest.getLabelCol());
        assertTrue(anovaTest.getFlatten());
    }

    @Test
    public void testOutputSchema() {
        ANOVATest anovaTest =
                new ANOVATest().setFeaturesCol("test_features").setLabelCol("test_label");
        Table output = anovaTest.transform(denseInputTable)[0];
        assertEquals(
                Arrays.asList("pValues", "degreesOfFreedom", "fValues"),
                output.getResolvedSchema().getColumnNames());

        anovaTest.setFlatten(true);
        output = anovaTest.transform(denseInputTable)[0];
        assertEquals(
                Arrays.asList("featureIndex", "pValue", "degreeOfFreedom", "fValue"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        ANOVATest anovaTest = new ANOVATest();

        Table denseOutput = anovaTest.transform(denseInputTable)[0];
        verifyTransformationResult(denseOutput, EXPECTED_OUTPUT_DENSE);

        Table sparseOutput = anovaTest.transform(sparseInputTable)[0];
        verifyTransformationResult(sparseOutput, EXPECTED_OUTPUT_SPARSE);
    }

    @Test
    public void testTransformWithFlatten() throws Exception {
        ANOVATest anovaTest = new ANOVATest().setFlatten(true);

        Table denseOutput = anovaTest.transform(denseInputTable)[0];
        verifyFlattenTransformationResult(denseOutput, EXPECTED_FLATTENED_OUTPUT_DENSE);

        Table sparseOutput = anovaTest.transform(sparseInputTable)[0];
        verifyFlattenTransformationResult(sparseOutput, EXPECTED_FLATTENED_OUTPUT_SPARSE);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        ANOVATest anovaTest = new ANOVATest();
        ANOVATest loadedANOVATest =
                TestUtils.saveAndReload(
                        tEnv, anovaTest, tempFolder.newFolder().getAbsolutePath(), ANOVATest::load);
        Table output = loadedANOVATest.transform(denseInputTable)[0];
        verifyTransformationResult(output, EXPECTED_OUTPUT_DENSE);
    }
}
