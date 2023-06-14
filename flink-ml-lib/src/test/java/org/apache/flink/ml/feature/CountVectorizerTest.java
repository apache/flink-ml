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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.ml.feature.countvectorizer.CountVectorizer;
import org.apache.flink.ml.feature.countvectorizer.CountVectorizerModel;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.test.util.TestBaseUtils;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Tests {@link CountVectorizer} and {@link CountVectorizerModel}. */
public class CountVectorizerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table inputTable;

    private static final double EPS = 1.0e-5;
    private static final List<Row> INPUT_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of((Object) new String[] {"a", "c", "b", "c"}),
                            Row.of((Object) new String[] {"c", "d", "e"}),
                            Row.of((Object) new String[] {"a", "b", "c"}),
                            Row.of((Object) new String[] {"e", "f"}),
                            Row.of((Object) new String[] {"a", "c", "a"})));

    private static final List<SparseIntDoubleVector> EXPECTED_OUTPUT =
            new ArrayList<>(
                    Arrays.asList(
                            Vectors.sparse(
                                    6,
                                    IntStream.of(0, 1, 2).toArray(),
                                    DoubleStream.of(2.0, 1.0, 1.0).toArray()),
                            Vectors.sparse(
                                    6,
                                    IntStream.of(0, 3, 4).toArray(),
                                    DoubleStream.of(1.0, 1.0, 1.0).toArray()),
                            Vectors.sparse(
                                    6,
                                    IntStream.of(0, 1, 2).toArray(),
                                    DoubleStream.of(1.0, 1.0, 1.0).toArray()),
                            Vectors.sparse(
                                    6,
                                    IntStream.of(3, 5).toArray(),
                                    DoubleStream.of(1.0, 1.0).toArray()),
                            Vectors.sparse(
                                    6,
                                    IntStream.of(0, 1).toArray(),
                                    DoubleStream.of(1.0, 2.0).toArray())));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        inputTable = tEnv.fromDataStream(env.fromCollection(INPUT_DATA)).as("input");
    }

    private static void verifyPredictionResult(
            Table output, String outputCol, List<SparseIntDoubleVector> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<SparseIntDoubleVector> stream =
                tEnv.toDataStream(output)
                        .map(
                                (MapFunction<Row, SparseIntDoubleVector>)
                                        row -> (SparseIntDoubleVector) row.getField(outputCol));
        List<SparseIntDoubleVector> result = IteratorUtils.toList(stream.executeAndCollect());
        TestBaseUtils.compareResultCollections(expected, result, TestUtils::compare);
    }

    @Test
    public void testParam() {
        CountVectorizer countVectorizer = new CountVectorizer();
        assertEquals("input", countVectorizer.getInputCol());
        assertEquals("output", countVectorizer.getOutputCol());
        assertEquals((double) Long.MAX_VALUE, countVectorizer.getMaxDF(), EPS);
        assertEquals(1.0, countVectorizer.getMinDF(), EPS);
        assertEquals(1.0, countVectorizer.getMinTF(), EPS);
        assertEquals(1 << 18, countVectorizer.getVocabularySize());
        assertFalse(countVectorizer.getBinary());

        countVectorizer
                .setInputCol("test_input")
                .setOutputCol("test_output")
                .setMinDF(0.1)
                .setMaxDF(0.9)
                .setMinTF(10)
                .setVocabularySize(1000)
                .setBinary(true);
        assertEquals("test_input", countVectorizer.getInputCol());
        assertEquals("test_output", countVectorizer.getOutputCol());
        assertEquals(0.9, countVectorizer.getMaxDF(), EPS);
        assertEquals(0.1, countVectorizer.getMinDF(), EPS);
        assertEquals(10, countVectorizer.getMinTF(), EPS);
        assertEquals(1000, countVectorizer.getVocabularySize());
        assertTrue(countVectorizer.getBinary());
    }

    @Test
    public void testInvalidMinMaxDF() {
        String errMessage = "maxDF must be >= minDF.";
        CountVectorizer countVectorizer = new CountVectorizer();
        countVectorizer.setMaxDF(0.1);
        countVectorizer.setMinDF(0.2);
        try {
            countVectorizer.fit(inputTable);
            fail();
        } catch (Throwable e) {
            assertEquals(errMessage, e.getMessage());
        }
        countVectorizer.setMaxDF(1);
        countVectorizer.setMinDF(2);
        try {
            countVectorizer.fit(inputTable);
            fail();
        } catch (Throwable e) {
            assertEquals(errMessage, e.getMessage());
        }
        countVectorizer.setMaxDF(1);
        countVectorizer.setMinDF(0.9);
        try {
            CountVectorizerModel model = countVectorizer.fit(inputTable);
            Table output = model.transform(inputTable)[0];
            output.execute().print();
            fail();
        } catch (Throwable e) {
            assertEquals(errMessage, ExceptionUtils.getRootCause(e).getMessage());
        }
        countVectorizer.setMaxDF(0.1);
        countVectorizer.setMinDF(10);
        try {
            CountVectorizerModel model = countVectorizer.fit(inputTable);
            Table output = model.transform(inputTable)[0];
            output.execute().print();
            fail();
        } catch (Throwable e) {
            assertEquals(errMessage, ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testOutputSchema() {
        CountVectorizer countVectorizer =
                new CountVectorizer().setInputCol("test_input").setOutputCol("test_output");
        CountVectorizerModel model = countVectorizer.fit(inputTable);
        Table output = model.transform(inputTable.as("test_input"))[0];
        assertEquals(
                Arrays.asList("test_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        CountVectorizer countVectorizer = new CountVectorizer();
        CountVectorizerModel model = countVectorizer.fit(inputTable);
        Table output = model.transform(inputTable)[0];

        verifyPredictionResult(output, countVectorizer.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        CountVectorizer countVectorizer = new CountVectorizer();
        CountVectorizer loadedCountVectorizer =
                TestUtils.saveAndReload(
                        tEnv,
                        countVectorizer,
                        tempFolder.newFolder().getAbsolutePath(),
                        CountVectorizer::load);
        CountVectorizerModel model = loadedCountVectorizer.fit(inputTable);
        CountVectorizerModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        CountVectorizerModel::load);
        assertEquals(
                Arrays.asList("vocabulary"),
                loadedModel.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = loadedModel.transform(inputTable)[0];
        verifyPredictionResult(output, countVectorizer.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testFitOnEmptyData() {
        Table emptyTable =
                tEnv.fromDataStream(env.fromCollection(INPUT_DATA).filter(x -> x.getArity() == 0))
                        .as("input");
        CountVectorizer countVectorizer = new CountVectorizer();
        CountVectorizerModel model = countVectorizer.fit(emptyTable);
        Table modelDataTable = model.getModelData()[0];
        try {
            modelDataTable.execute().print();
            fail();
        } catch (Throwable e) {
            assertEquals("The training set is empty.", ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testMinMaxDF() throws Exception {
        List<SparseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.sparse(
                                        4,
                                        IntStream.of(0, 1, 2).toArray(),
                                        DoubleStream.of(2.0, 1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        4,
                                        IntStream.of(0, 3).toArray(),
                                        DoubleStream.of(1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        4,
                                        IntStream.of(0, 1, 2).toArray(),
                                        DoubleStream.of(1.0, 1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        4,
                                        IntStream.of(3).toArray(),
                                        DoubleStream.of(1.0).toArray()),
                                Vectors.sparse(
                                        4,
                                        IntStream.of(0, 1).toArray(),
                                        DoubleStream.of(1.0, 2.0).toArray())));
        CountVectorizer countVectorizer = new CountVectorizer().setMinDF(2).setMaxDF(4);
        CountVectorizerModel model = countVectorizer.fit(inputTable);
        Table output = model.transform(inputTable)[0];
        verifyPredictionResult(output, countVectorizer.getOutputCol(), expectedOutput);

        countVectorizer.setMinDF(0.4).setMaxDF(0.8);
        model = countVectorizer.fit(inputTable);
        output = model.transform(inputTable)[0];
        verifyPredictionResult(output, countVectorizer.getOutputCol(), expectedOutput);
    }

    @Test
    public void testMinTF() throws Exception {
        List<SparseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.sparse(
                                        6,
                                        IntStream.of(0).toArray(),
                                        DoubleStream.of(2.0).toArray()),
                                Vectors.sparse(6, new int[0], new double[0]),
                                Vectors.sparse(6, new int[0], new double[0]),
                                Vectors.sparse(
                                        6,
                                        IntStream.of(3, 5).toArray(),
                                        DoubleStream.of(1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        6,
                                        IntStream.of(1).toArray(),
                                        DoubleStream.of(2.0).toArray())));
        CountVectorizer countVectorizer = new CountVectorizer().setMinTF(0.5);
        CountVectorizerModel model = countVectorizer.fit(inputTable);
        Table output = model.transform(inputTable)[0];
        verifyPredictionResult(output, countVectorizer.getOutputCol(), expectedOutput);
    }

    @Test
    public void testBinary() throws Exception {
        List<SparseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.sparse(
                                        6,
                                        IntStream.of(0, 1, 2).toArray(),
                                        DoubleStream.of(1.0, 1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        6,
                                        IntStream.of(0, 3, 4).toArray(),
                                        DoubleStream.of(1.0, 1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        6,
                                        IntStream.of(0, 1, 2).toArray(),
                                        DoubleStream.of(1.0, 1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        6,
                                        IntStream.of(3, 5).toArray(),
                                        DoubleStream.of(1.0, 1.0).toArray()),
                                Vectors.sparse(
                                        6,
                                        IntStream.of(0, 1).toArray(),
                                        DoubleStream.of(1.0, 1.0).toArray())));
        CountVectorizer countVectorizer = new CountVectorizer().setBinary(true);
        CountVectorizerModel model = countVectorizer.fit(inputTable);
        Table output = model.transform(inputTable)[0];
        verifyPredictionResult(output, countVectorizer.getOutputCol(), expectedOutput);
    }

    @Test
    public void testVocabularySize() throws Exception {
        List<SparseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.sparse(
                                        2,
                                        IntStream.of(0, 1).toArray(),
                                        DoubleStream.of(2.0, 1.0).toArray()),
                                Vectors.sparse(
                                        2,
                                        IntStream.of(0).toArray(),
                                        DoubleStream.of(1.0).toArray()),
                                Vectors.sparse(
                                        2,
                                        IntStream.of(0, 1).toArray(),
                                        DoubleStream.of(1.0, 1.0).toArray()),
                                Vectors.sparse(2, new int[0], new double[0]),
                                Vectors.sparse(
                                        2,
                                        IntStream.of(0, 1).toArray(),
                                        DoubleStream.of(1.0, 2.0).toArray())));
        CountVectorizer countVectorizer = new CountVectorizer().setVocabularySize(2);
        CountVectorizerModel model = countVectorizer.fit(inputTable);
        Table output = model.transform(inputTable)[0];
        verifyPredictionResult(output, countVectorizer.getOutputCol(), expectedOutput);
    }

    @Test
    public void testGetModelData() throws Exception {
        CountVectorizer countVectorizer = new CountVectorizer();
        CountVectorizerModel model = countVectorizer.fit(inputTable);
        Table modelData = model.getModelData()[0];
        assertEquals(Arrays.asList("vocabulary"), modelData.getResolvedSchema().getColumnNames());

        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        String[] vocabulary = (String[]) modelRows.get(0).getField(0);
        String[] expectedVocabulary = {"c", "a", "b", "e", "d", "f"};
        assertArrayEquals(expectedVocabulary, vocabulary);
    }

    @Test
    public void testSetModelData() throws Exception {
        CountVectorizer countVectorizer = new CountVectorizer();
        CountVectorizerModel modelA = countVectorizer.fit(inputTable);
        Table modelData = modelA.getModelData()[0];
        CountVectorizerModel modelB = new CountVectorizerModel().setModelData(modelData);
        Table output = modelB.transform(inputTable)[0];
        verifyPredictionResult(output, countVectorizer.getOutputCol(), EXPECTED_OUTPUT);
    }
}
