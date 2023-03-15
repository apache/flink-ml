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

import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.stats.chisqtest.ChiSqTest;
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
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Tests the {@link ChiSqTest}. */
public class ChiSqTestTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private Table inputTableWithDoubleLabel;
    private Table inputTableWithIntegerLabel;

    private final List<Row> samplesWithDoubleLabel =
            Arrays.asList(
                    Row.of(0., Vectors.dense(5, 1.)),
                    Row.of(2., Vectors.dense(6, 2.)),
                    Row.of(1., Vectors.dense(7, 2.)),
                    Row.of(1., Vectors.dense(5, 4.)),
                    Row.of(0., Vectors.dense(5, 1.)),
                    Row.of(2., Vectors.dense(6, 2.)),
                    Row.of(1., Vectors.dense(7, 2.)),
                    Row.of(1., Vectors.dense(5, 4.)),
                    Row.of(2., Vectors.dense(5, 1.)),
                    Row.of(0., Vectors.dense(5, 2.)),
                    Row.of(0., Vectors.dense(5, 2.)),
                    Row.of(1., Vectors.dense(9, 4.)),
                    Row.of(1., Vectors.dense(9, 3.)));

    private final List<Row> expectedChiSqTestResultWithDoubleLabel =
            Collections.singletonList(
                    Row.of(
                            Vectors.dense(0.03419350755, 0.24220177737),
                            new int[] {6, 6},
                            Vectors.dense(13.61904761905, 7.94444444444)));

    private final List<Row> expectedChiSqTestResultWithDoubleLabelWithFlatten =
            Arrays.asList(
                    Row.of(0, 0.03419350755, 6, 13.61904761905),
                    Row.of(1, 0.24220177737, 6, 7.94444444444));

    private final List<Row> samplesWithIntegerLabel =
            Arrays.asList(
                    Row.of(33, Vectors.dense(5, 0)),
                    Row.of(44, Vectors.dense(6, 1)),
                    Row.of(55, Vectors.dense(7, 1)),
                    Row.of(11, Vectors.dense(5, 1)),
                    Row.of(11, Vectors.dense(5, 0)),
                    Row.of(33, Vectors.dense(6, 2)),
                    Row.of(22, Vectors.dense(7, 2)),
                    Row.of(66, Vectors.dense(5, 3)),
                    Row.of(77, Vectors.dense(5, 3)),
                    Row.of(88, Vectors.dense(5, 4)),
                    Row.of(77, Vectors.dense(5, 6)),
                    Row.of(44, Vectors.dense(9, 6)),
                    Row.of(11, Vectors.dense(9, 8)));

    private final List<Row> expectedChiSqTestResultWithIntegerLabel =
            Collections.singletonList(
                    Row.of(
                            Vectors.dense(0.35745138256, 0.39934987096),
                            new int[] {21, 42},
                            Vectors.dense(22.75, 43.69444444444)));

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        inputTableWithDoubleLabel =
                tEnv.fromDataStream(env.fromCollection(samplesWithDoubleLabel))
                        .as("label", "features");
        inputTableWithIntegerLabel =
                tEnv.fromDataStream(env.fromCollection(samplesWithIntegerLabel))
                        .as("label", "features");
    }

    private static void verifyPredictionResult(Table output, List<Row> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Row> outputDataStream = tEnv.toDataStream(output);

        List<Row> result = IteratorUtils.toList(outputDataStream.executeAndCollect());

        compareResultCollections(
                expected,
                result,
                (row1, row2) -> {
                    if (!row1.equals(row2)) {
                        return 1;
                    } else {
                        return 0;
                    }
                });
    }

    @Test
    public void testParam() {
        ChiSqTest chiSqTest = new ChiSqTest();
        assertEquals("label", chiSqTest.getLabelCol());
        assertEquals("features", chiSqTest.getFeaturesCol());
        assertFalse(chiSqTest.getFlatten());

        chiSqTest.setLabelCol("test_label").setFeaturesCol("test_features").setFlatten(true);

        assertEquals("test_features", chiSqTest.getFeaturesCol());
        assertEquals("test_label", chiSqTest.getLabelCol());
        assertTrue(chiSqTest.getFlatten());
    }

    @Test
    public void testOutputSchema() {
        ChiSqTest chiSqTest = new ChiSqTest().setFeaturesCol("features").setLabelCol("label");

        Table output = chiSqTest.transform(inputTableWithDoubleLabel)[0];
        assertEquals(
                Arrays.asList("pValues", "degreesOfFreedom", "statistics"),
                output.getResolvedSchema().getColumnNames());

        chiSqTest.setFlatten(true);

        output = chiSqTest.transform(inputTableWithDoubleLabel)[0];
        assertEquals(
                Arrays.asList("featureIndex", "pValue", "degreeOfFreedom", "statistic"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        ChiSqTest chiSqTest = new ChiSqTest().setFeaturesCol("features").setLabelCol("label");

        Table output1 = chiSqTest.transform(inputTableWithDoubleLabel)[0];
        verifyPredictionResult(output1, expectedChiSqTestResultWithDoubleLabel);

        Table output2 = chiSqTest.transform(inputTableWithIntegerLabel)[0];
        verifyPredictionResult(output2, expectedChiSqTestResultWithIntegerLabel);
    }

    @Test
    public void testTransformWithFlatten() throws Exception {
        ChiSqTest chiSqTest =
                new ChiSqTest().setFlatten(true).setFeaturesCol("features").setLabelCol("label");

        Table output1 = chiSqTest.transform(inputTableWithDoubleLabel)[0];
        verifyPredictionResult(output1, expectedChiSqTestResultWithDoubleLabelWithFlatten);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        ChiSqTest chiSqTest = new ChiSqTest().setFeaturesCol("features").setLabelCol("label");

        ChiSqTest loadedChiSqTest =
                TestUtils.saveAndReload(
                        tEnv, chiSqTest, tempFolder.newFolder().getAbsolutePath(), ChiSqTest::load);
        Table output1 = loadedChiSqTest.transform(inputTableWithDoubleLabel)[0];
        verifyPredictionResult(output1, expectedChiSqTestResultWithDoubleLabel);
    }
}
