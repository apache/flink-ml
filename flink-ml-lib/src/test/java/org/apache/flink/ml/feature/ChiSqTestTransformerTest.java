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

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.feature.chisqtest.ChiSqTestTransformer;
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

/** Tests {@link ChiSqTestTransformerTest}. */
public class ChiSqTestTransformerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private Table inputTable;

    private final List<Row> samples =
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

    private final List<Row> expectedChiSqTestResult =
            Arrays.asList(
                    Row.of("f1", 0.03419350755, 13.61904761905, 6),
                    Row.of("f2", 0.24220177737, 7.94444444444, 6));

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        inputTable = tEnv.fromDataStream(env.fromCollection(samples)).as("label", "f1", "f2");
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
        ChiSqTestTransformer chiSqTest = new ChiSqTestTransformer();

        chiSqTest.setInputCols("f1", "f2").setLabelCol("label");

        assertArrayEquals(new String[] {"f1", "f2"}, chiSqTest.getInputCols());
        assertEquals("label", chiSqTest.getLabelCol());
    }

    @Test
    public void testOutputSchema() {
        ChiSqTestTransformer chiSqTest =
                new ChiSqTestTransformer().setInputCols("f1", "f2").setLabelCol("label");

        Table output = chiSqTest.transform(inputTable)[0];
        assertEquals(
                Arrays.asList("column", "pValue", "statistic", "degreesOfFreedom"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        ChiSqTestTransformer chiSqTest =
                new ChiSqTestTransformer().setInputCols("f1", "f2").setLabelCol("label");

        Table output = chiSqTest.transform(inputTable)[0];
        verifyPredictionResult(output, expectedChiSqTestResult);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        ChiSqTestTransformer chiSqTest =
                new ChiSqTestTransformer().setInputCols("f1", "f2").setLabelCol("label");

        ChiSqTestTransformer loadedBucketizer =
                TestUtils.saveAndReload(tEnv, chiSqTest, tempFolder.newFolder().getAbsolutePath());
        Table output = loadedBucketizer.transform(inputTable)[0];
        verifyPredictionResult(output, expectedChiSqTestResult);
    }
}
