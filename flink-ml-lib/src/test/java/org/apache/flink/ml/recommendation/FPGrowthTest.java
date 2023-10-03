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

package org.apache.flink.ml.recommendation;

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.recommendation.fpgrowth.FPGrowth;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/** Tests {@link FPGrowth}. */
public class FPGrowthTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private static final int defaultParallelism = 4;
    private static StreamExecutionEnvironment env;
    private static StreamTableEnvironment tEnv;
    private Table inputTable;

    List<Row> expectedPatterns =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of("B", 5L, 1L),
                            Row.of("C", 4L, 1L),
                            Row.of("B,C", 4L, 2L),
                            Row.of("A", 3L, 1L),
                            Row.of("B,A", 3L, 2L),
                            Row.of("C,A", 3L, 2L),
                            Row.of("B,C,A", 3L, 3L),
                            Row.of("E", 3L, 1L),
                            Row.of("B,E", 3L, 2L),
                            Row.of("D", 3L, 1L),
                            Row.of("B,D", 3L, 2L)));
    List<Row> expectedRules =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of("B=>E", 2L, 1.0, 0.6, 0.6, 3L),
                            Row.of("B=>C", 2L, 1.0, 0.8, 0.8, 4L),
                            Row.of("A=>B", 2L, 1.0, 0.6, 1.0, 3L),
                            Row.of("B=>A", 2L, 1.0, 0.6, 0.6, 3L),
                            Row.of("A=>C", 2L, 1.25, 0.6, 1.0, 3L),
                            Row.of("D=>B", 2L, 1.0, 0.6, 1.0, 3L),
                            Row.of("B=>D", 2L, 1.0, 0.6, 0.6, 3L),
                            Row.of("C,A=>B", 3L, 1.0, 0.6, 1.0, 3L),
                            Row.of("B,A=>C", 3L, 1.25, 0.6, 1.0, 3L),
                            Row.of("E=>B", 2L, 1.0, 0.6, 1.0, 3L),
                            Row.of("C=>B", 2L, 1.0, 0.8, 1.0, 4L),
                            Row.of("C=>A", 2L, 1.25, 0.6, 0.75, 3L),
                            Row.of("B,C=>A", 3L, 1.25, 0.6, 0.75, 3L)));

    public void checkResult(List<Row> expected, CloseableIterator<Row> result) {
        List<Row> actual = new ArrayList<>();
        while (result.hasNext()) {
            Row row = result.next();
            actual.add(row);
        }

        expected.sort(
                (o1, o2) -> {
                    String s1 = o1.getFieldAs(0);
                    String s2 = o2.getFieldAs(0);
                    return s1.compareTo(s2);
                });

        actual.sort(
                (o1, o2) -> {
                    String s1 = o1.getFieldAs(0);
                    String s2 = o2.getFieldAs(0);
                    return s1.compareTo(s2);
                });

        Assert.assertArrayEquals(expected.toArray(), actual.toArray());
    }

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        env.getConfig().setParallelism(defaultParallelism);
        tEnv = StreamTableEnvironment.create(env);
        List<Row> inputRows =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(""),
                                Row.of("A,B,C,D"),
                                Row.of("B,C,E"),
                                Row.of("A,B,C,E"),
                                Row.of("B,D,E"),
                                Row.of("A,B,C,D,A")));

        inputTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                inputRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {BasicTypeInfo.STRING_TYPE_INFO},
                                        new String[] {"transactions"})));
    }

    @Test
    public void testParam() {
        FPGrowth fpGrowth = new FPGrowth();
        assertEquals("items", fpGrowth.getItemsCol());
        assertEquals(",", fpGrowth.getFieldDelimiter());
        assertEquals(10, fpGrowth.getMaxPatternLength());
        assertEquals(0.6, fpGrowth.getMinConfidence(), 1e-9);
        assertEquals(0.02, fpGrowth.getMinSupport(), 1e-9);
        assertEquals(-1, fpGrowth.getMinSupportCount());
        assertEquals(1.0, fpGrowth.getMinLift(), 1e-9);

        fpGrowth.setItemsCol("transactions")
                .setFieldDelimiter(" ")
                .setMaxPatternLength(100)
                .setMinLift(0.5)
                .setMinConfidence(0.3)
                .setMinSupport(0.3)
                .setMinSupportCount(10);

        assertEquals("transactions", fpGrowth.getItemsCol());
        assertEquals(" ", fpGrowth.getFieldDelimiter());
        assertEquals(100, fpGrowth.getMaxPatternLength());
        assertEquals(0.3, fpGrowth.getMinConfidence(), 1e-9);
        assertEquals(0.3, fpGrowth.getMinSupport(), 1e-9);
        assertEquals(10, fpGrowth.getMinSupportCount());
        assertEquals(0.5, fpGrowth.getMinLift(), 1e-9);
    }

    @Test
    public void testTransform() {
        FPGrowth fpGrowth = new FPGrowth().setItemsCol("transactions").setMinSupport(0.6);
        Table[] results = fpGrowth.transform(inputTable);
        CloseableIterator<Row> patterns = results[0].execute().collect();
        checkResult(expectedPatterns, patterns);
        CloseableIterator<Row> rules = results[1].execute().collect();
        checkResult(expectedRules, rules);
    }

    @Test
    public void testOutputSchema() {
        FPGrowth fpGrowth = new FPGrowth().setItemsCol("transactions").setMinSupportCount(3);
        Table[] results = fpGrowth.transform(inputTable);
        assertEquals(
                Arrays.asList("items", "support_count", "item_count"),
                results[0].getResolvedSchema().getColumnNames());
        assertEquals(
                Arrays.asList(
                        "rule",
                        "item_count",
                        "lift",
                        "support_percent",
                        "confidence_percent",
                        "transaction_count"),
                results[1].getResolvedSchema().getColumnNames());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        FPGrowth fpGrowth = new FPGrowth().setItemsCol("transactions").setMinSupportCount(3);
        FPGrowth loadedFPGrowth =
                TestUtils.saveAndReload(
                        tEnv, fpGrowth, tempFolder.newFolder().getAbsolutePath(), FPGrowth::load);
        Table[] results = loadedFPGrowth.transform(inputTable);
        CloseableIterator<Row> patterns = results[0].execute().collect();
        checkResult(expectedPatterns, patterns);
        CloseableIterator<Row> rules = results[1].execute().collect();
        checkResult(expectedRules, rules);
    }
}
