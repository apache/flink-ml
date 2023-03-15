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

import org.apache.flink.ml.feature.regextokenizer.RegexTokenizer;
import org.apache.flink.ml.feature.regextokenizer.RegexTokenizerParams;
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

/** Tests {@link RegexTokenizer}. */
public class RegexTokenizerTest extends AbstractTestBase {
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table inputDataTable;

    private static final List<Row> INPUT =
            Arrays.asList(Row.of("Test for tokenization."), Row.of("Te,st. punct"));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT);
        inputDataTable = tEnv.fromDataStream(dataStream).as("input");
    }

    @Test
    public void testParam() {
        RegexTokenizer regexTokenizer = new RegexTokenizer();
        assertEquals("input", regexTokenizer.getInputCol());
        assertEquals("output", regexTokenizer.getOutputCol());
        assertEquals(1, regexTokenizer.getMinTokenLength());
        assertEquals(true, regexTokenizer.getGaps());
        assertEquals("\\s+", regexTokenizer.getPattern());
        assertEquals(true, regexTokenizer.getToLowercase());

        regexTokenizer
                .setInputCol("testInputCol")
                .setOutputCol("testOutputCol")
                .setMinTokenLength(3)
                .setGaps(false)
                .setPattern("\\s")
                .setToLowercase(false);

        assertEquals("testInputCol", regexTokenizer.getInputCol());
        assertEquals("testOutputCol", regexTokenizer.getOutputCol());
        assertEquals(3, regexTokenizer.getMinTokenLength());
        assertEquals(false, regexTokenizer.getGaps());
        assertEquals("\\s", regexTokenizer.getPattern());
        assertEquals(false, regexTokenizer.getToLowercase());
    }

    @Test
    public void testOutputSchema() {
        RegexTokenizer regexTokenizer = new RegexTokenizer();
        inputDataTable =
                tEnv.fromDataStream(env.fromElements(Row.of("", ""))).as("input", "dummyInput");
        Table output = regexTokenizer.transform(inputDataTable)[0];
        assertEquals(
                Arrays.asList(
                        regexTokenizer.getInputCol(), "dummyInput", regexTokenizer.getOutputCol()),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        List<Row> expectedRows;
        int minTokenLength = RegexTokenizerParams.MIN_TOKEN_LENGTH.defaultValue;
        boolean gaps = RegexTokenizerParams.GAPS.defaultValue;
        String pattern = RegexTokenizerParams.PATTERN.defaultValue;
        boolean toLowercase = RegexTokenizerParams.TO_LOWERCASE.defaultValue;

        // default option.
        expectedRows =
                Arrays.asList(
                        Row.of((Object) new String[] {"test", "for", "tokenization."}),
                        Row.of((Object) new String[] {"te,st.", "punct"}));
        checkTransform(minTokenLength, gaps, pattern, toLowercase, expectedRows);

        // default option except toLowercase = false.
        expectedRows =
                Arrays.asList(
                        Row.of((Object) new String[] {"Test", "for", "tokenization."}),
                        Row.of((Object) new String[] {"Te,st.", "punct"}));
        toLowercase = false;
        checkTransform(minTokenLength, gaps, pattern, toLowercase, expectedRows);

        // default option except gaps = false, pattern = "\\w+|\\p{Punct}".
        expectedRows =
                Arrays.asList(
                        Row.of((Object) new String[] {"test", "for", "tokenization", "."}),
                        Row.of((Object) new String[] {"te", ",", "st", ".", "punct"}));
        gaps = false;
        pattern = "\\w+|\\p{Punct}";
        toLowercase = true;
        checkTransform(minTokenLength, gaps, pattern, toLowercase, expectedRows);

        // default option except gaps = false, minTokenLength = 3, pattern = "\\w+|\\p{Punct}".
        gaps = false;
        minTokenLength = 3;
        pattern = "\\w+|\\p{Punct}";
        expectedRows =
                Arrays.asList(
                        Row.of((Object) new String[] {"test", "for", "tokenization"}),
                        Row.of((Object) new String[] {"punct"}));
        checkTransform(minTokenLength, gaps, pattern, toLowercase, expectedRows);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        RegexTokenizer regexTokenizer = new RegexTokenizer();
        List<Row> expectedRows =
                Arrays.asList(
                        Row.of((Object) new String[] {"test", "for", "tokenization."}),
                        Row.of((Object) new String[] {"te,st.", "punct"}));
        RegexTokenizer loadedRegexTokenizer =
                TestUtils.saveAndReload(
                        tEnv,
                        regexTokenizer,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        RegexTokenizer::load);
        Table output = loadedRegexTokenizer.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedRegexTokenizer.getOutputCol(), expectedRows);
    }

    private void checkTransform(
            int minTokenLength,
            boolean gaps,
            String pattern,
            boolean toLowercase,
            List<Row> expectedOutput)
            throws Exception {
        RegexTokenizer regexTokenizer =
                new RegexTokenizer()
                        .setMinTokenLength(minTokenLength)
                        .setGaps(gaps)
                        .setPattern(pattern)
                        .setToLowercase(toLowercase);
        Table output = regexTokenizer.transform(inputDataTable)[0];
        verifyOutputResult(output, regexTokenizer.getOutputCol(), expectedOutput);
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
