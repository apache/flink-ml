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

import org.apache.flink.ml.feature.tokenizer.Tokenizer;
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

/** Tests {@link Tokenizer}. */
public class TokenizerTest extends AbstractTestBase {
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table inputDataTable;

    private static final List<Row> INPUT =
            Arrays.asList(Row.of("Test for tokenization."), Row.of("Te,st. punct"));

    private static final List<Row> EXPECTED_OUTPUT =
            Arrays.asList(
                    Row.of((Object) new String[] {"test", "for", "tokenization."}),
                    Row.of((Object) new String[] {"te,st.", "punct"}));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT);
        inputDataTable = tEnv.fromDataStream(dataStream).as("input");
    }

    @Test
    public void testParam() {
        Tokenizer tokenizer = new Tokenizer();
        assertEquals("input", tokenizer.getInputCol());
        assertEquals("output", tokenizer.getOutputCol());

        tokenizer.setInputCol("testInputCol").setOutputCol("testOutputCol");
        assertEquals("testInputCol", tokenizer.getInputCol());
        assertEquals("testOutputCol", tokenizer.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        Tokenizer tokenizer = new Tokenizer();
        inputDataTable =
                tEnv.fromDataStream(env.fromElements(Row.of("", ""))).as("input", "dummyInput");
        Table output = tokenizer.transform(inputDataTable)[0];
        assertEquals(
                Arrays.asList(tokenizer.getInputCol(), "dummyInput", tokenizer.getOutputCol()),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        Tokenizer tokenizer = new Tokenizer();
        Table output = tokenizer.transform(inputDataTable)[0];
        verifyOutputResult(output, tokenizer.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        Tokenizer tokenizer = new Tokenizer();
        Tokenizer loadedTokenizer =
                TestUtils.saveAndReload(
                        tEnv,
                        tokenizer,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        Tokenizer::load);
        Table output = loadedTokenizer.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedTokenizer.getOutputCol(), EXPECTED_OUTPUT);
    }

    private void verifyOutputResult(Table output, String outputCol, List<Row> expectedOutput)
            throws Exception {
        DataStream<Row> dataStream = tEnv.toDataStream(output.select(Expressions.$(outputCol)));
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(expectedOutput.size(), results.size());
        results.sort(Comparator.comparingInt(o -> ((String[]) o.getField(0))[0].hashCode()));
        expectedOutput.sort(Comparator.comparingInt(o -> ((String[]) o.getField(0))[0].hashCode()));
        for (int i = 0; i < expectedOutput.size(); i++) {
            assertArrayEquals(
                    (String[]) results.get(i).getField(0),
                    (String[]) expectedOutput.get(i).getField(0));
        }
    }
}
