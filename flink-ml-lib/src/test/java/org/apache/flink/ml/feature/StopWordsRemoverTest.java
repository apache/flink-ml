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

import org.apache.flink.ml.builder.PipelineModel;
import org.apache.flink.ml.feature.stopwordsremover.StopWordsRemover;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.ValidationException;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Tests {@link StopWordsRemover}. */
public class StopWordsRemoverTest extends AbstractTestBase {
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
    }

    private static void verifyOutputResult(StopWordsRemover remover, Table inputTable) {
        Table outputTable = remover.transform(inputTable)[0];

        int expectedLength = IteratorUtils.toList(inputTable.execute().collect()).size();
        int count = 0;
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();
            String[] expected = row.getFieldAs("expected");
            String[] actual = row.getFieldAs("filtered");
            assertArrayEquals(expected, actual);
            count++;
        }
        assertEquals(expectedLength, count);
    }

    @Test
    public void testParams() {
        StopWordsRemover remover = new StopWordsRemover();
        assertTrue(Arrays.asList(remover.getStopWords()).containsAll(Arrays.asList("i", "would")));
        assertTrue(
                Arrays.asList(Locale.US.toString(), Locale.getDefault().toString())
                        .contains(remover.getLocale()));
        assertFalse(remover.getCaseSensitive());

        remover.setInputCols("f1", "f2")
                .setOutputCols("o1", "o2")
                .setStopWords(StopWordsRemover.loadDefaultStopWords("turkish"))
                .setLocale(Locale.US.toString())
                .setCaseSensitive(true);

        assertArrayEquals(new String[] {"f1", "f2"}, remover.getInputCols());
        assertArrayEquals(new String[] {"o1", "o2"}, remover.getOutputCols());
        assertTrue(
                Arrays.asList(remover.getStopWords()).containsAll(Arrays.asList("acaba", "yani")));
        assertEquals(Locale.US.toString(), remover.getLocale());
        assertTrue(remover.getCaseSensitive());
    }

    @Test
    public void testOutputSchema() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"test", "test"}, new String[] {"test", "test"}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover().setInputCols("raw").setOutputCols("filtered");

        Table outputTable = remover.transform(inputTable)[0];

        assertEquals(
                Arrays.asList("raw", "expected", "filtered"),
                outputTable.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"test", "test"}, new String[] {"test", "test"}),
                        Row.of(new String[] {"a", "b", "c", "d"}, new String[] {"b", "c", "d"}),
                        Row.of(new String[] {"a", "the", "an"}, new String[] {}),
                        Row.of(new String[] {"A", "The", "AN"}, new String[] {}),
                        Row.of(new String[] {null}, new String[] {null}),
                        Row.of(new String[] {}, new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover().setInputCols("raw").setOutputCols("filtered");

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testTransformWithStopWordsList() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"test", "test"}, new String[] {}),
                        Row.of(new String[] {"a", "b", "c", "d"}, new String[] {"b", "c", "d"}),
                        Row.of(new String[] {"a", "the", "an"}, new String[] {}),
                        Row.of(new String[] {"A", "The", "AN"}, new String[] {}),
                        Row.of(new String[] {null}, new String[] {}),
                        Row.of(new String[] {}, new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw")
                        .setOutputCols("filtered")
                        .setStopWords(new String[] {"test", "a", "an", "the", null});

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testTransformWithLocaledInputCaseInsensitive() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"mİlk", "and", "nuts"}, new String[] {"and", "nuts"}),
                        Row.of(
                                new String[] {"cookIe", "and", "nuts"},
                                new String[] {"cookIe", "and", "nuts"}),
                        Row.of(new String[] {null}, new String[] {null}),
                        Row.of(new String[] {}, new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw")
                        .setOutputCols("filtered")
                        .setStopWords(new String[] {"milk", "cookie"})
                        .setCaseSensitive(false)
                        .setLocale("tr");

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testTransformWithLocaledInputCaseSensitive() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(
                                new String[] {"mİlk", "and", "nuts"},
                                new String[] {"mİlk", "and", "nuts"}),
                        Row.of(
                                new String[] {"cookIe", "and", "nuts"},
                                new String[] {"cookIe", "and", "nuts"}),
                        Row.of(new String[] {null}, new String[] {null}),
                        Row.of(new String[] {}, new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw")
                        .setOutputCols("filtered")
                        .setStopWords(new String[] {"milk", "cookie"})
                        .setCaseSensitive(true)
                        .setLocale("tr");

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testInvalidLocale() {
        try {
            new StopWordsRemover().setLocale("rt");
            fail();
        } catch (Exception e) {
            assertEquals(IllegalArgumentException.class, e.getClass());
            assertEquals("Parameter locale is given an invalid value rt", e.getMessage());
        }
    }

    @Test
    public void testAvailableLocales() {
        assertTrue(StopWordsRemover.getAvailableLocales().contains(Locale.US.toString()));

        StopWordsRemover remover = new StopWordsRemover();
        for (String locale : StopWordsRemover.getAvailableLocales()) {
            remover.setLocale(locale);
        }
    }

    @Test
    public void testTransformCaseSensitive() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"A"}, new String[] {"A"}),
                        Row.of(new String[] {"The", "the"}, new String[] {"The"}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw")
                        .setOutputCols("filtered")
                        .setCaseSensitive(true);

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testDefaultStopWordsOfSupportedLanguagesNotEmtpy() {
        List<String> supportedLanguages =
                Arrays.asList(
                        "danish",
                        "dutch",
                        "english",
                        "finnish",
                        "french",
                        "german",
                        "hungarian",
                        "italian",
                        "norwegian",
                        "portuguese",
                        "russian",
                        "spanish",
                        "swedish",
                        "turkish");
        for (String language : supportedLanguages) {
            assertTrue(StopWordsRemover.loadDefaultStopWords(language).length > 0);
        }
    }

    @Test
    public void testTransformWithLanguageSelection() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"acaba", "ama", "biri"}, new String[] {}),
                        Row.of(new String[] {"hep", "her", "scala"}, new String[] {"scala"}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw")
                        .setOutputCols("filtered")
                        .setStopWords(StopWordsRemover.loadDefaultStopWords("turkish"));

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testTransformWithIgnoredWords() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(
                                new String[] {"python", "scala", "a"},
                                new String[] {"python", "scala", "a"}),
                        Row.of(
                                new String[] {"Python", "Scala", "swift"},
                                new String[] {"Python", "Scala", "swift"}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        Set<String> stopWords =
                new HashSet<>(Arrays.asList(StopWordsRemover.loadDefaultStopWords("english")));
        stopWords.remove("a");

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw")
                        .setOutputCols("filtered")
                        .setStopWords(stopWords.toArray(new String[0]));

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testTransformWithAdditionalWords() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"python", "scala", "a"}, new String[] {}),
                        Row.of(new String[] {"Python", "Scala", "swift"}, new String[] {"swift"}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        Set<String> stopWords =
                new HashSet<>(Arrays.asList(StopWordsRemover.loadDefaultStopWords("english")));
        stopWords.addAll(Arrays.asList("python", "scala"));

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw")
                        .setOutputCols("filtered")
                        .setStopWords(stopWords.toArray(new String[0]));

        verifyOutputResult(remover, inputTable);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"test", "test"}, new String[] {"test", "test"}),
                        Row.of(new String[] {"a", "b", "c", "d"}, new String[] {"b", "c", "d"}),
                        Row.of(new String[] {"a", "the", "an"}, new String[] {}),
                        Row.of(new String[] {"A", "The", "AN"}, new String[] {}),
                        Row.of(new String[] {null}, new String[] {null}),
                        Row.of(new String[] {}, new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");

        StopWordsRemover remover =
                new StopWordsRemover().setInputCols("raw").setOutputCols("filtered");

        StopWordsRemover loadedRemover =
                TestUtils.saveAndReload(
                        tEnv,
                        remover,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        StopWordsRemover::load);

        verifyOutputResult(loadedRemover, inputTable);
    }

    @Test
    public void testOutputColumnAlreadyExists() {
        try {
            DataStream<Row> inputStream =
                    env.fromElements(
                            Row.of(new String[] {"The", "the", "swift"}, new String[] {"swift"}));
            Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");
            StopWordsRemover remover =
                    new StopWordsRemover().setInputCols("raw").setOutputCols("expected");
            remover.transform(inputTable);
            fail();
        } catch (Exception e) {
            assertEquals(ValidationException.class, e.getClass());
            assertEquals("Ambiguous column name: expected", e.getMessage());
        }
    }

    @Test
    public void testTransformMultipleColumns() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(
                                new String[] {"test", "test"},
                                new String[] {"test1", "test2"},
                                new String[] {},
                                new String[] {"test1", "test2"}),
                        Row.of(
                                new String[] {"a", "b", "c", "d"},
                                new String[] {"a", "b"},
                                new String[] {"b", "c", "d"},
                                new String[] {"b"}),
                        Row.of(
                                new String[] {"a", "the", "an"},
                                new String[] {"a", "the", "test1"},
                                new String[] {},
                                new String[] {"test1"}),
                        Row.of(
                                new String[] {"A", "The", "AN"},
                                new String[] {"A", "The", "AN"},
                                new String[] {},
                                new String[] {}),
                        Row.of(
                                new String[] {null},
                                new String[] {null},
                                new String[] {null},
                                new String[] {null}),
                        Row.of(new String[] {}, new String[] {}, new String[] {}, new String[] {}));
        Table inputTable =
                tEnv.fromDataStream(inputStream).as("raw1", "raw2", "expected1", "expected2");

        StopWordsRemover remover =
                new StopWordsRemover()
                        .setInputCols("raw1", "raw2")
                        .setOutputCols("filtered1", "filtered2")
                        .setStopWords(new String[] {"test", "a", "an", "the"});

        Table outputTable = remover.transform(inputTable)[0];

        int expectedLength = IteratorUtils.toList(inputTable.execute().collect()).size();
        int count = 0;
        for (CloseableIterator<Row> it = outputTable.execute().collect(); it.hasNext(); ) {
            Row row = it.next();

            String[] expected = row.getFieldAs("expected1");
            String[] actual = row.getFieldAs("filtered1");
            assertArrayEquals(expected, actual);

            expected = row.getFieldAs("expected2");
            actual = row.getFieldAs("filtered2");
            assertArrayEquals(expected, actual);

            count++;
        }
        assertEquals(expectedLength, count);
    }

    @Test
    public void testCompareSingleMultipleRemoverInPipeline() {
        DataStream<Row> inputStream =
                env.fromElements(
                        Row.of(new String[] {"test", "test"}, new String[] {"test1", "test2"}),
                        Row.of(new String[] {"a", "b", "c", "d"}, new String[] {"a", "b"}),
                        Row.of(new String[] {"a", "the", "an"}, new String[] {"a", "the", "test1"}),
                        Row.of(new String[] {"A", "The", "AN"}, new String[] {"A", "The", "AN"}),
                        Row.of(new String[] {null}, new String[] {null}),
                        Row.of(new String[] {}, new String[] {}));
        Table inputTable = tEnv.fromDataStream(inputStream).as("input1", "input2");

        PipelineModel multiColsPipeline =
                new PipelineModel(
                        Collections.singletonList(
                                new StopWordsRemover()
                                        .setInputCols("input1", "input2")
                                        .setOutputCols("output1", "output2")));

        PipelineModel singleColPipeline =
                new PipelineModel(
                        Arrays.asList(
                                new StopWordsRemover()
                                        .setInputCols("input1")
                                        .setOutputCols("output1"),
                                new StopWordsRemover()
                                        .setInputCols("input2")
                                        .setOutputCols("output2")));

        assertEquals(
                new HashSet<Row>(
                        IteratorUtils.toList(
                                multiColsPipeline.transform(inputTable)[0].execute().collect())),
                new HashSet<Row>(
                        IteratorUtils.toList(
                                singleColPipeline.transform(inputTable)[0].execute().collect())));
    }

    @Test
    public void testMismatchInputOutputCols() {
        try {
            DataStream<Row> inputStream =
                    env.fromElements(
                            Row.of(new String[] {"The", "the", "swift"}, new String[] {"swift"}));
            Table inputTable = tEnv.fromDataStream(inputStream).as("raw", "expected");
            StopWordsRemover remover =
                    new StopWordsRemover()
                            .setInputCols("raw")
                            .setOutputCols("expected1", "expected2");
            remover.transform(inputTable);
            fail();
        } catch (Exception e) {
            assertEquals(IllegalArgumentException.class, e.getClass());
            assertNull(e.getMessage());
        }
    }
}
