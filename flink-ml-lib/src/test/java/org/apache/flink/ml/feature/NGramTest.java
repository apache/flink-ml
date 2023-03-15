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

import org.apache.flink.ml.feature.ngram.NGram;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Expressions;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link NGram}. */
public class NGramTest extends AbstractTestBase {
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table inputDataTable;
    private static final List<Row> EXPECTED_OUTPUT =
            Arrays.asList(
                    Row.of((Object) new String[0]),
                    Row.of((Object) new String[] {"a b", "b c"}),
                    Row.of((Object) new String[] {"a b", "b c", "c d"}));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        List<Row> input =
                Arrays.asList(
                        Row.of((Object) new String[0]),
                        Row.of((Object) new String[] {"a", "b", "c"}),
                        Row.of((Object) new String[] {"a", "b", "c", "d"}));
        DataStream<Row> dataStream = env.fromCollection(input);
        inputDataTable = tEnv.fromDataStream(dataStream).as("input");
    }

    @Test
    public void testParam() {
        NGram nGram = new NGram();
        assertEquals("input", nGram.getInputCol());
        assertEquals("output", nGram.getOutputCol());
        assertEquals(2, nGram.getN());

        nGram.setInputCol("testInputCol").setOutputCol("testOutputCol").setN(5);
        assertEquals("testInputCol", nGram.getInputCol());
        assertEquals("testOutputCol", nGram.getOutputCol());
        assertEquals(5, nGram.getN());
    }

    @Test
    public void testOutputSchema() {
        NGram nGram = new NGram();
        inputDataTable =
                tEnv.fromDataStream(env.fromElements(Row.of(new String[] {""}, "")))
                        .as("input", "dummyInput");
        Table output = nGram.transform(inputDataTable)[0];
        assertEquals(
                Arrays.asList(nGram.getInputCol(), "dummyInput", nGram.getOutputCol()),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        NGram nGram = new NGram();
        Table output = nGram.transform(inputDataTable)[0];
        verifyOutputResult(output, nGram.getOutputCol());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        NGram nGram = new NGram();
        NGram loadedNGram =
                TestUtils.saveAndReload(
                        tEnv, nGram, TEMPORARY_FOLDER.newFolder().getAbsolutePath(), NGram::load);
        Table output = loadedNGram.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedNGram.getOutputCol());
    }

    private void verifyOutputResult(Table output, String outputCol) throws Exception {
        DataStream<Row> dataStream = tEnv.toDataStream(output.select(Expressions.$(outputCol)));
        List<Row> actualResults = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(EXPECTED_OUTPUT.size(), actualResults.size());
        actualResults.sort(Comparator.comparingInt(o -> ((String[]) o.getField(0)).length));
        for (int i = 0; i < EXPECTED_OUTPUT.size(); i++) {
            assertArrayEquals(
                    (String[]) EXPECTED_OUTPUT.get(i).getField(0),
                    (String[]) actualResults.get(i).getField(0));
        }
    }
}
