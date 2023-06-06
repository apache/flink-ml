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
import org.apache.flink.ml.stats.fvaluetest.FValueTest;
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

/** Tests the {@link FValueTest}. */
public class FValueTestTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private Table denseInputTable;
    private Table sparseInputTable;

    private static final double EPS = 1.0e-5;
    private static final List<Row> DENSE_INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            0.19775997,
                            Vectors.dense(
                                    0.15266373,
                                    0.30235661,
                                    0.06203641,
                                    0.45986034,
                                    0.83525338,
                                    0.92699705)),
                    Row.of(
                            0.66009772,
                            Vectors.dense(
                                    0.72698898,
                                    0.76849622,
                                    0.26920507,
                                    0.64402929,
                                    0.09337326,
                                    0.07968589)),
                    Row.of(
                            0.80865842,
                            Vectors.dense(
                                    0.58961375,
                                    0.34334054,
                                    0.98887615,
                                    0.62647321,
                                    0.68177928,
                                    0.55225681)),
                    Row.of(
                            0.34142582,
                            Vectors.dense(
                                    0.26886006,
                                    0.37325939,
                                    0.2229281,
                                    0.1864426,
                                    0.39064809,
                                    0.19316241)),
                    Row.of(
                            0.84756607,
                            Vectors.dense(
                                    0.61091093,
                                    0.88280845,
                                    0.62233882,
                                    0.25311894,
                                    0.17993031,
                                    0.81640447)),
                    Row.of(
                            0.53360225,
                            Vectors.dense(
                                    0.22537162,
                                    0.51685714,
                                    0.51849582,
                                    0.60037494,
                                    0.53262048,
                                    0.01331005)),
                    Row.of(
                            0.90053371,
                            Vectors.dense(
                                    0.52409726,
                                    0.89588471,
                                    0.76990129,
                                    0.1228517,
                                    0.29587269,
                                    0.61202358)),
                    Row.of(
                            0.78779561,
                            Vectors.dense(
                                    0.72613812,
                                    0.46349747,
                                    0.76911037,
                                    0.19163103,
                                    0.55786672,
                                    0.55077816)),
                    Row.of(
                            0.51604647,
                            Vectors.dense(
                                    0.47222549,
                                    0.79188496,
                                    0.11524968,
                                    0.6813039,
                                    0.36233361,
                                    0.34420889)),
                    Row.of(
                            0.35325637,
                            Vectors.dense(
                                    0.44951875,
                                    0.02694226,
                                    0.41524769,
                                    0.9222317,
                                    0.09120557,
                                    0.31512178)),
                    Row.of(
                            0.51408926,
                            Vectors.dense(
                                    0.52802224,
                                    0.32806203,
                                    0.44891554,
                                    0.01633442,
                                    0.0970269,
                                    0.69258857)),
                    Row.of(
                            0.84489897,
                            Vectors.dense(
                                    0.83594341,
                                    0.42432199,
                                    0.8487743,
                                    0.54679121,
                                    0.35410346,
                                    0.72724968)),
                    Row.of(
                            0.55342816,
                            Vectors.dense(
                                    0.09385168,
                                    0.8928588,
                                    0.33625828,
                                    0.89183268,
                                    0.296849,
                                    0.30164829)),
                    Row.of(
                            0.89405683,
                            Vectors.dense(
                                    0.80624061,
                                    0.83760997,
                                    0.63428133,
                                    0.3113273,
                                    0.02944858,
                                    0.39977732)),
                    Row.of(
                            0.54588131,
                            Vectors.dense(
                                    0.51817346,
                                    0.00738845,
                                    0.77494778,
                                    0.8544712,
                                    0.13153282,
                                    0.28767364)),
                    Row.of(
                            0.96038024,
                            Vectors.dense(
                                    0.32658881,
                                    0.90655956,
                                    0.99955954,
                                    0.77088429,
                                    0.04284752,
                                    0.96525111)),
                    Row.of(
                            0.71349698,
                            Vectors.dense(
                                    0.97521246,
                                    0.2025168,
                                    0.67985305,
                                    0.46534506,
                                    0.92001748,
                                    0.72820735)),
                    Row.of(
                            0.43456735,
                            Vectors.dense(
                                    0.24585653,
                                    0.01953996,
                                    0.70598881,
                                    0.77448287,
                                    0.4729746,
                                    0.80146736)),
                    Row.of(
                            0.52462506,
                            Vectors.dense(
                                    0.17539792,
                                    0.72016934,
                                    0.3678759,
                                    0.53209295,
                                    0.29719397,
                                    0.37429151)),
                    Row.of(
                            0.43074793,
                            Vectors.dense(
                                    0.72810013,
                                    0.39850784,
                                    0.1058295,
                                    0.39858265,
                                    0.52196395,
                                    0.1060125)));

    private static final List<Row> SPARSE_INPUT_DATA =
            Arrays.asList(
                    Row.of(4.6, Vectors.dense(6.0, 7.0, 0.0, 7.0, 6.0, 0.0, 0.0).toSparse()),
                    Row.of(6.6, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0, 0.0).toSparse()),
                    Row.of(5.1, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0, 0.0).toSparse()),
                    Row.of(7.6, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0, 0.0).toSparse()),
                    Row.of(9.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0, 0.0).toSparse()),
                    Row.of(
                            9.0,
                            Vectors.dense(Double.NaN, 9.0, 6.0, 4.0, 0.0, 0.0, 0.0).toSparse()));

    private static final Row EXPECTED_OUTPUT_DENSE =
            Row.of(
                    Vectors.dense(
                            1.73658700e-02,
                            1.49916659e-02,
                            1.12697153e-04,
                            4.26990301e-01,
                            2.75911201e-01,
                            1.93549275e-01),
                    new long[] {18, 18, 18, 18, 18, 18},
                    Vectors.dense(
                            6.86260598,
                            7.23175589,
                            24.11424725,
                            0.6605354,
                            1.26266286,
                            1.82421406));

    private static final List<Row> EXPECTED_FLATTENED_OUTPUT_DENSE =
            Arrays.asList(
                    Row.of(0, 1.73658700e-02, 18, 6.86260598),
                    Row.of(1, 1.49916659e-02, 18, 7.23175589),
                    Row.of(2, 1.12697153e-04, 18, 24.11424725),
                    Row.of(3, 4.26990301e-01, 18, 0.6605354),
                    Row.of(4, 2.75911201e-01, 18, 1.26266286),
                    Row.of(5, 1.93549275e-01, 18, 1.82421406));

    private static final Row EXPECTED_OUTPUT_SPARSE =
            Row.of(
                    Vectors.dense(
                            Double.NaN,
                            0.19167161,
                            0.06506426,
                            0.75183662,
                            0.16111045,
                            0.89090362,
                            Double.NaN),
                    new long[] {4, 4, 4, 4, 4, 4, 4},
                    Vectors.dense(
                            Double.NaN,
                            2.46254817,
                            6.37164347,
                            0.1147488,
                            2.94816821,
                            0.02134755,
                            Double.NaN));

    private static final List<Row> EXPECTED_FLATTENED_OUTPUT_SPARSE =
            Arrays.asList(
                    Row.of(0, Double.NaN, 4, Double.NaN),
                    Row.of(1, 0.19167161, 4, 2.46254817),
                    Row.of(2, 0.06506426, 4, 6.37164347),
                    Row.of(3, 0.75183662, 4, 0.1147488),
                    Row.of(4, 0.16111045, 4, 2.94816821),
                    Row.of(5, 0.89090362, 4, 0.02134755),
                    Row.of(6, Double.NaN, 4, Double.NaN));

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
        FValueTest fValueTest = new FValueTest();
        assertEquals("label", fValueTest.getLabelCol());
        assertEquals("features", fValueTest.getFeaturesCol());
        assertFalse(fValueTest.getFlatten());

        fValueTest.setLabelCol("test_label").setFeaturesCol("test_features").setFlatten(true);

        assertEquals("test_features", fValueTest.getFeaturesCol());
        assertEquals("test_label", fValueTest.getLabelCol());
        assertTrue(fValueTest.getFlatten());
    }

    @Test
    public void testOutputSchema() {
        FValueTest fValueTest =
                new FValueTest().setFeaturesCol("test_features").setLabelCol("test_label");
        Table output = fValueTest.transform(denseInputTable)[0];
        assertEquals(
                Arrays.asList("pValues", "degreesOfFreedom", "fValues"),
                output.getResolvedSchema().getColumnNames());

        fValueTest.setFlatten(true);
        output = fValueTest.transform(denseInputTable)[0];
        assertEquals(
                Arrays.asList("featureIndex", "pValue", "degreeOfFreedom", "fValue"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        FValueTest fValueTest = new FValueTest();

        Table denseOutput = fValueTest.transform(denseInputTable)[0];
        verifyTransformationResult(denseOutput, EXPECTED_OUTPUT_DENSE);

        Table sparseOutput = fValueTest.transform(sparseInputTable)[0];
        verifyTransformationResult(sparseOutput, EXPECTED_OUTPUT_SPARSE);
    }

    @Test
    public void testTransformWithFlatten() throws Exception {
        FValueTest fValueTest = new FValueTest().setFlatten(true);

        Table denseOutput = fValueTest.transform(denseInputTable)[0];
        verifyFlattenTransformationResult(denseOutput, EXPECTED_FLATTENED_OUTPUT_DENSE);

        Table sparseOutput = fValueTest.transform(sparseInputTable)[0];
        verifyFlattenTransformationResult(sparseOutput, EXPECTED_FLATTENED_OUTPUT_SPARSE);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        FValueTest fValueTest = new FValueTest();
        FValueTest loadedFValueTest =
                TestUtils.saveAndReload(
                        tEnv,
                        fValueTest,
                        tempFolder.newFolder().getAbsolutePath(),
                        FValueTest::load);
        Table output = loadedFValueTest.transform(denseInputTable)[0];
        verifyTransformationResult(output, EXPECTED_OUTPUT_DENSE);
    }
}
