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
import org.apache.flink.ml.recommendation.swing.Swing;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link Swing}. */
public class SwingTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table inputTable;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        List<Row> inputRows =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(0L, 10L),
                                Row.of(0L, 11L),
                                Row.of(0L, 12L),
                                Row.of(1L, 13L),
                                Row.of(1L, 12L),
                                Row.of(2L, 10L),
                                Row.of(2L, 11L),
                                Row.of(2L, 12L),
                                Row.of(3L, 13L),
                                Row.of(3L, 12L),
                                Row.of(4L, 12L),
                                Row.of(4L, 10L),
                                Row.of(4L, 11L),
                                Row.of(4L, 12L),
                                Row.of(4L, 13L)));
        inputTable =
                tEnv.fromDataStream(
                        env.fromCollection(
                                inputRows,
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            BasicTypeInfo.LONG_TYPE_INFO,
                                            BasicTypeInfo.LONG_TYPE_INFO
                                        },
                                        new String[] {"user", "item"})));
    }

    private void compareResultAndExpected(List<Row> results) {
        List<Row> expectedScoreRows =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(10L, "11,0.058845768947156235;12,0.058845768947156235"),
                                Row.of(11L, "10,0.058845768947156235;12,0.058845768947156235"),
                                Row.of(
                                        12L,
                                        "13,0.09134833828228624;10,0.058845768947156235;11,0.058845768947156235"),
                                Row.of(13L, "12,0.09134833828228624")));

        results.sort(Comparator.comparing(o -> o.getFieldAs(0)));

        for (int i = 0; i < results.size(); i++) {
            Row result = results.get(i);
            String itemRankScore = result.getFieldAs(1);
            Row expect = expectedScoreRows.get(i);
            assertEquals(expect.getField(0), result.getField(0));
            assertEquals(expect.getField(1), itemRankScore);
        }
    }

    @Test
    public void testParam() {
        Swing swing = new Swing();

        assertEquals("item", swing.getItemCol());
        assertEquals("user", swing.getUserCol());
        assertEquals(100, swing.getK());
        assertEquals(1000, swing.getMaxUserNumPerItem());
        assertEquals(10, swing.getMinUserBehavior());
        assertEquals(1000, swing.getMaxUserBehavior());
        assertEquals(15, swing.getAlpha1());
        assertEquals(0, swing.getAlpha2());
        assertEquals(0.3, swing.getBeta(), 1e-9);

        swing.setItemCol("item_1")
                .setUserCol("user_1")
                .setK(20)
                .setMaxUserNumPerItem(500)
                .setMinUserBehavior(10)
                .setMaxUserBehavior(50)
                .setAlpha1(5)
                .setAlpha2(1)
                .setBeta(0.35);

        assertEquals("item_1", swing.getItemCol());
        assertEquals("user_1", swing.getUserCol());
        assertEquals(20, swing.getK());
        assertEquals(500, swing.getMaxUserNumPerItem());
        assertEquals(10, swing.getMinUserBehavior());
        assertEquals(50, swing.getMaxUserBehavior());
        assertEquals(5, swing.getAlpha1());
        assertEquals(1, swing.getAlpha2());
        assertEquals(0.35, swing.getBeta(), 1e-9);
    }

    @Test
    public void testInputWithIllegalDataType() {
        List<Row> rows =
                new ArrayList<>(Arrays.asList(Row.of(0, "10"), Row.of(1, "11"), Row.of(2, "")));

        DataStream<Row> dataStream =
                env.fromCollection(
                        rows,
                        new RowTypeInfo(
                                new TypeInformation[] {
                                    BasicTypeInfo.LONG_TYPE_INFO, BasicTypeInfo.STRING_TYPE_INFO
                                },
                                new String[] {"user", "item"}));
        Table data = tEnv.fromDataStream(dataStream);

        try {
            Table[] swingResultTables = new Swing().setMinUserBehavior(1).transform(data);
            swingResultTables[0].execute().collect();
            fail();
        } catch (RuntimeException e) {
            assertEquals(IllegalArgumentException.class, e.getClass());
            assertEquals("The types of user and item must be Long.", e.getMessage());
        }
    }

    @Test
    public void testInputWithNull() {
        List<Row> rows =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(0L, 10L),
                                Row.of(null, 12L),
                                Row.of(1L, 13L),
                                Row.of(3L, 12L)));

        DataStream<Row> dataStream =
                env.fromCollection(
                        rows,
                        new RowTypeInfo(
                                new TypeInformation[] {
                                    BasicTypeInfo.LONG_TYPE_INFO, BasicTypeInfo.LONG_TYPE_INFO
                                },
                                new String[] {"user", "item"}));
        Table data = tEnv.fromDataStream(dataStream);
        Swing swing = new Swing().setMinUserBehavior(1);
        Table[] swingResultTables = swing.transform(data);

        try {
            swingResultTables[0].execute().collect().next();
            fail();
        } catch (RuntimeException e) {
            Throwable exception = ExceptionUtils.getRootCause(e);
            assertEquals(RuntimeException.class, exception.getClass());
            assertEquals("Data of user and item column must not be null.", exception.getMessage());
        }
    }

    @Test
    public void testOutputSchema() {
        Swing swing = new Swing().setOutputCol("item_score").setMinUserBehavior(1);
        Table[] swingResultTables = swing.transform(inputTable);
        Table output = swingResultTables[0];

        assertEquals(
                Arrays.asList("item", "item_score"), output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() {
        Swing swing = new Swing().setMinUserBehavior(2).setMaxUserBehavior(3);
        Table[] swingResultTables = swing.transform(inputTable);
        Table outputTable = swingResultTables[0];
        List<Row> results = IteratorUtils.toList(outputTable.execute().collect());
        compareResultAndExpected(results);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        Swing swing = new Swing().setMinUserBehavior(1);
        Swing loadedSwing =
                TestUtils.saveAndReload(
                        tEnv, swing, tempFolder.newFolder().getAbsolutePath(), Swing::load);
        Table outputTable = loadedSwing.transform(inputTable)[0];
        List<Row> results = IteratorUtils.toList(outputTable.execute().collect());
        compareResultAndExpected(results);
    }
}
