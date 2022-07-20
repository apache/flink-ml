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

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.stats.chisqtest.ChiSqTest;
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
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link ChiSqTestTest}. */
public class ChiSqTestTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private Table inputTableWithDoubleLabel;
    private Table inputTableWithIntegerLabel;
    private Table inputTableWithStringLabel;

    private final List<Row> samplesWithDoubleLabel =
            Arrays.asList(
                    Row.of(0., 5, 1.),
                    Row.of(2., 6, 2.),
                    Row.of(1., 7, 2.),
                    Row.of(1., 5, 4.),
                    Row.of(0., 5, 1.),
                    Row.of(2., 6, 2.),
                    Row.of(1., 7, 2.),
                    Row.of(1., 5, 4.),
                    Row.of(2., 5, 1.),
                    Row.of(0., 5, 2.),
                    Row.of(0., 5, 2.),
                    Row.of(1., 9, 4.),
                    Row.of(1., 9, 3.));

    private final List<Row> expectedChiSqTestResultWithDoubleLabel =
            Arrays.asList(
                    Row.of("f1", 0.03419350755, 13.61904761905, 6),
                    Row.of("f2", 0.24220177737, 7.94444444444, 6));

    private final List<Row> samplesWithIntegerLabel =
            Arrays.asList(
                    Row.of(33, 5, "a"),
                    Row.of(44, 6, "b"),
                    Row.of(55, 7, "b"),
                    Row.of(11, 5, "b"),
                    Row.of(11, 5, "a"),
                    Row.of(33, 6, "c"),
                    Row.of(22, 7, "c"),
                    Row.of(66, 5, "d"),
                    Row.of(77, 5, "d"),
                    Row.of(88, 5, "f"),
                    Row.of(77, 5, "h"),
                    Row.of(44, 9, "h"),
                    Row.of(11, 9, "j"));

    private final List<Row> expectedChiSqTestResultWithIntegerLabel =
            Arrays.asList(
                    Row.of("f1", 0.35745138256, 22.75, 21),
                    Row.of("f2", 0.39934987096, 43.69444444444, 42));

    private final List<Row> samplesWithStringLabel =
            Arrays.asList(
                    Row.of("v1", 11, 21.22),
                    Row.of("v1", 33, 22.33),
                    Row.of("v2", 22, 32.44),
                    Row.of("v3", 11, 54.22),
                    Row.of("v3", 33, 22.22),
                    Row.of("v2", 22, 22.22),
                    Row.of("v4", 55, 22.22),
                    Row.of("v5", 11, 41.11),
                    Row.of("v6", 55, 14.41),
                    Row.of("v7", 13, 25.55),
                    Row.of("v8", 14, 25.55),
                    Row.of("v9", 14, 44.44),
                    Row.of("v9", 14, 31.11));

    private final List<Row> expectedChiSqTestResultWithStringLabel =
            Arrays.asList(
                    Row.of("f1", 0.06672255089, 54.16666666667, 40),
                    Row.of("f2", 0.42335512313, 73.66666666667, 72));

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        inputTableWithDoubleLabel =
                tEnv.fromDataStream(env.fromCollection(samplesWithDoubleLabel))
                        .as("label", "f1", "f2");
        inputTableWithIntegerLabel =
                tEnv.fromDataStream(env.fromCollection(samplesWithIntegerLabel))
                        .as("label", "f1", "f2");
        inputTableWithStringLabel =
                tEnv.fromDataStream(env.fromCollection(samplesWithStringLabel))
                        .as("label", "f1", "f2");
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

        chiSqTest.setInputCols("f1", "f2").setLabelCol("label");

        assertArrayEquals(new String[] {"f1", "f2"}, chiSqTest.getInputCols());
        assertEquals("label", chiSqTest.getLabelCol());
    }

    @Test
    public void testOutputSchema() {
        ChiSqTest chiSqTest = new ChiSqTest().setInputCols("f1", "f2").setLabelCol("label");

        Table output = chiSqTest.transform(inputTableWithDoubleLabel)[0];
        assertEquals(
                Arrays.asList("column", "pValue", "statistic", "degreesOfFreedom"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        ChiSqTest chiSqTest = new ChiSqTest().setInputCols("f1", "f2").setLabelCol("label");

        Table output1 = chiSqTest.transform(inputTableWithDoubleLabel)[0];
        verifyPredictionResult(output1, expectedChiSqTestResultWithDoubleLabel);

        Table output2 = chiSqTest.transform(inputTableWithIntegerLabel)[0];
        verifyPredictionResult(output2, expectedChiSqTestResultWithIntegerLabel);

        Table output3 = chiSqTest.transform(inputTableWithStringLabel)[0];
        verifyPredictionResult(output3, expectedChiSqTestResultWithStringLabel);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        ChiSqTest chiSqTest = new ChiSqTest().setInputCols("f1", "f2").setLabelCol("label");

        ChiSqTest loadedBucketizer =
                TestUtils.saveAndReload(tEnv, chiSqTest, tempFolder.newFolder().getAbsolutePath());
        Table output1 = loadedBucketizer.transform(inputTableWithDoubleLabel)[0];
        verifyPredictionResult(output1, expectedChiSqTestResultWithDoubleLabel);

        Table output2 = loadedBucketizer.transform(inputTableWithIntegerLabel)[0];
        verifyPredictionResult(output2, expectedChiSqTestResultWithIntegerLabel);

        Table output3 = loadedBucketizer.transform(inputTableWithStringLabel)[0];
        verifyPredictionResult(output3, expectedChiSqTestResultWithStringLabel);
    }
}
