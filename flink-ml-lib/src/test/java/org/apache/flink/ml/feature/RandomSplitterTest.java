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

import org.apache.flink.ml.feature.randomsplitter.RandomSplitter;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStreamSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link RandomSplitter}. */
public class RandomSplitterTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        env.setParallelism(1);
        tEnv = StreamTableEnvironment.create(env);
    }

    private Table getTable(int size) {
        DataStreamSource<Long> dataStream = env.fromSequence(0L, size);
        return tEnv.fromDataStream(dataStream);
    }

    @Test
    public void testParam() {
        RandomSplitter splitter = new RandomSplitter();
        splitter.setWeights(0.3, 0.4).setSeed(5L);
        assertArrayEquals(new Double[] {0.3, 0.4}, splitter.getWeights());
        assertEquals(5L, splitter.getSeed());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable =
                tEnv.fromDataStream(env.fromElements(Row.of("", "")))
                        .as("test_input", "dummy_input");

        RandomSplitter splitter = new RandomSplitter().setWeights(0.5, 0.1);
        Table[] output = splitter.transform(tempTable);
        assertEquals(2, output.length);
        for (Table table : output) {
            assertEquals(
                    Arrays.asList("test_input", "dummy_input"),
                    table.getResolvedSchema().getColumnNames());
        }
    }

    @Test
    public void testWeights() throws Exception {
        Table data = getTable(1000);
        RandomSplitter splitter = new RandomSplitter().setWeights(2.0, 1.0, 2.0);
        Table[] output = splitter.transform(data);

        List<Row> result0 = IteratorUtils.toList(tEnv.toDataStream(output[0]).executeAndCollect());
        List<Row> result1 = IteratorUtils.toList(tEnv.toDataStream(output[1]).executeAndCollect());
        List<Row> result2 = IteratorUtils.toList(tEnv.toDataStream(output[2]).executeAndCollect());
        assertEquals(result0.size() / 400.0, 1.0, 0.1);
        assertEquals(result1.size() / 200.0, 1.0, 0.1);
        assertEquals(result2.size() / 400.0, 1.0, 0.1);
        verifyResultTables(data, output);
    }

    @Test
    public void testSeed() throws Exception {
        Table data = getTable(100);
        RandomSplitter splitter = new RandomSplitter().setWeights(2.0, 1.0, 2.0);

        Table[] output0 = splitter.transform(data);
        List<Row> result00 =
                IteratorUtils.toList(tEnv.toDataStream(output0[0]).executeAndCollect());
        List<Row> result01 =
                IteratorUtils.toList(tEnv.toDataStream(output0[1]).executeAndCollect());
        List<Row> result02 =
                IteratorUtils.toList(tEnv.toDataStream(output0[2]).executeAndCollect());

        Table[] output1 = splitter.transform(data);
        List<Row> result10 =
                IteratorUtils.toList(tEnv.toDataStream(output1[0]).executeAndCollect());
        List<Row> result11 =
                IteratorUtils.toList(tEnv.toDataStream(output1[1]).executeAndCollect());
        List<Row> result12 =
                IteratorUtils.toList(tEnv.toDataStream(output1[2]).executeAndCollect());

        assertEquals(result00.size(), result10.size());
        assertEquals(result01.size(), result11.size());
        assertEquals(result02.size(), result12.size());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        Table data = getTable(2000);
        RandomSplitter randomSplitter = new RandomSplitter().setWeights(4.0, 6.0);

        RandomSplitter splitterLoad =
                TestUtils.saveAndReload(
                        tEnv,
                        randomSplitter,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        RandomSplitter::load);

        Table[] output = splitterLoad.transform(data);
        List<Row> result0 = IteratorUtils.toList(tEnv.toDataStream(output[0]).executeAndCollect());
        List<Row> result1 = IteratorUtils.toList(tEnv.toDataStream(output[1]).executeAndCollect());
        assertEquals(result0.size() / 800.0, 1.0, 0.1);
        assertEquals(result1.size() / 1200.0, 1.0, 0.1);
        verifyResultTables(data, output);
    }

    private void verifyResultTables(Table input, Table[] output) throws Exception {
        List<Row> expectedData = IteratorUtils.toList(tEnv.toDataStream(input).executeAndCollect());
        List<Row> results = new ArrayList<>();
        for (Table table : output) {
            List<Row> result = IteratorUtils.toList(tEnv.toDataStream(table).executeAndCollect());
            results.addAll(result);
        }
        assertEquals(expectedData.size(), results.size());
        compareResultCollections(
                expectedData, results, Comparator.comparingLong(row -> row.getFieldAs(0)));
    }
}
