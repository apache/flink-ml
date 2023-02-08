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

package org.apache.flink.ml.feature.stringindexer;

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Tests the {@link StringIndexer} and {@link StringIndexerModel}. */
public class StringIndexerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainTable;
    private Table predictTable;

    private final String[][] expectedAlphabeticAscModelData =
            new String[][] {{"a", "b", "c", "d"}, {"-1.0", "0.0", "1.0", "2.0"}};
    private final List<Row> expectedAlphabeticAscPredictData =
            Arrays.asList(
                    Row.of("a", 2.0, 0.0, 3.0),
                    Row.of("b", 1.0, 1.0, 2.0),
                    Row.of("e", 2.0, 4.0, 3.0),
                    Row.of("f", null, 4.0, 4.0),
                    Row.of(null, null, 4.0, 4.0));
    private final List<Row> expectedAlphabeticDescPredictData =
            Arrays.asList(
                    Row.of("a", 2.0, 3.0, 0.0),
                    Row.of("b", 1.0, 2.0, 1.0),
                    Row.of("e", 2.0, 4.0, 0.0),
                    Row.of("f", null, 4.0, 4.0),
                    Row.of(null, null, 4.0, 4.0));
    private final List<Row> expectedFreqAscPredictData =
            Arrays.asList(
                    Row.of("a", 2.0, 2.0, 3.0),
                    Row.of("b", 1.0, 3.0, 1.0),
                    Row.of("e", 2.0, 4.0, 3.0),
                    Row.of("f", null, 4.0, 4.0),
                    Row.of(null, null, 4.0, 4.0));
    private final List<Row> expectedFreqDescPredictData =
            Arrays.asList(
                    Row.of("a", 2.0, 1.0, 0.0),
                    Row.of("b", 1.0, 0.0, 2.0),
                    Row.of("e", 2.0, 4.0, 0.0),
                    Row.of("f", null, 4.0, 4.0),
                    Row.of(null, null, 4.0, 4.0));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        List<Row> trainData =
                Arrays.asList(
                        Row.of("a", 1.0),
                        Row.of("b", 1.0),
                        Row.of("b", 2.0),
                        Row.of("c", 0.0),
                        Row.of("d", 2.0),
                        Row.of("a", 2.0),
                        Row.of("b", 2.0),
                        Row.of("b", -1.0),
                        Row.of("a", -1.0),
                        Row.of("c", -1.0),
                        Row.of("d", null),
                        Row.of(null, 2.0),
                        Row.of(null, null));
        trainTable =
                tEnv.fromDataStream(env.fromCollection(trainData)).as("inputCol1", "inputCol2");

        List<Row> predictData =
                Arrays.asList(
                        Row.of("a", 2.0),
                        Row.of("b", 1.0),
                        Row.of("e", 2.0),
                        Row.of("f", null),
                        Row.of(null, null));
        predictTable =
                tEnv.fromDataStream(env.fromCollection(predictData)).as("inputCol1", "inputCol2");
    }

    @Test
    public void testParam() {
        StringIndexer stringIndexer = new StringIndexer();
        assertEquals(stringIndexer.getStringOrderType(), StringIndexerParams.ARBITRARY_ORDER);
        assertEquals(stringIndexer.getHandleInvalid(), StringIndexerParams.ERROR_INVALID);

        stringIndexer
                .setInputCols("inputCol1", "inputCol2")
                .setOutputCols("outputCol1", "outputCol2")
                .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
                .setHandleInvalid(StringIndexerParams.SKIP_INVALID);

        assertArrayEquals(new String[] {"inputCol1", "inputCol2"}, stringIndexer.getInputCols());
        assertArrayEquals(new String[] {"outputCol1", "outputCol2"}, stringIndexer.getOutputCols());
        assertEquals(stringIndexer.getStringOrderType(), StringIndexerParams.ALPHABET_ASC_ORDER);
        assertEquals(stringIndexer.getHandleInvalid(), StringIndexerParams.SKIP_INVALID);
    }

    @Test
    public void testOutputSchema() {
        StringIndexer stringIndexer =
                new StringIndexer()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
                        .setHandleInvalid(StringIndexerParams.SKIP_INVALID);
        Table output = stringIndexer.fit(trainTable).transform(predictTable)[0];

        assertEquals(
                Arrays.asList("inputCol1", "inputCol2", "outputCol1", "outputCol2"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    @SuppressWarnings("all")
    public void testStringOrderType() throws Exception {
        StringIndexer stringIndexer =
                new StringIndexer()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setHandleInvalid(StringIndexerParams.KEEP_INVALID);
        Table output;
        List<Row> predictedResult;

        // AlphabetAsc order.
        stringIndexer.setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER);
        output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        predictedResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedAlphabeticAscPredictData, predictedResult);

        // AlphabetDesc order.
        stringIndexer.setStringOrderType(StringIndexerParams.ALPHABET_DESC_ORDER);
        output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        predictedResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedAlphabeticDescPredictData, predictedResult);

        // FrequencyAsc order.
        stringIndexer.setStringOrderType(StringIndexerParams.FREQUENCY_ASC_ORDER);
        output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        predictedResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedFreqAscPredictData, predictedResult);

        // FrequencyDesc order.
        stringIndexer.setStringOrderType(StringIndexerParams.FREQUENCY_DESC_ORDER);
        output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        predictedResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedFreqDescPredictData, predictedResult);

        // Arbitrary order.
        stringIndexer.setStringOrderType(StringIndexerParams.ARBITRARY_ORDER);
        output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        predictedResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        Set<Double> distinctStringsCol1 = new HashSet<>();
        Set<Double> distinctStringsCol2 = new HashSet<>();
        double index;
        for (Row r : predictedResult) {
            index = (Double) r.getField(2);
            distinctStringsCol1.add(index);
            assertTrue(index >= 0 && index <= 4);
            index = (Double) r.getField(3);
            assertTrue(index >= 0 && index <= 4);
            distinctStringsCol2.add(index);
        }

        assertEquals(3, distinctStringsCol1.size());
        assertEquals(3, distinctStringsCol2.size());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testHandleInvalid() throws Exception {
        StringIndexer stringIndexer =
                new StringIndexer()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER);

        Table output;
        List<Row> expectedResult;

        // Keeps invalid data.
        stringIndexer.setHandleInvalid(StringIndexerParams.KEEP_INVALID);
        output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        List<Row> predictedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedAlphabeticAscPredictData, predictedResult);

        // Skips invalid data.
        stringIndexer.setHandleInvalid(StringIndexerParams.SKIP_INVALID);
        output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        predictedResult = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        expectedResult = Arrays.asList(Row.of("a", 2.0, 0.0, 3.0), Row.of("b", 1.0, 1.0, 2.0));
        verifyPredictionResult(expectedResult, predictedResult);

        // Throws an exception on invalid data.
        stringIndexer.setHandleInvalid(StringIndexerParams.ERROR_INVALID);
        try {
            output = stringIndexer.fit(trainTable).transform(predictTable)[0];
            IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
            fail();
        } catch (Throwable e) {
            List<String> expectedMessages =
                    Stream.of("e", "f", "null")
                            .map(
                                    d ->
                                            String.format(
                                                    "The input contains unseen string: %s. See %s parameter for more options.",
                                                    d, HasHandleInvalid.HANDLE_INVALID))
                            .collect(Collectors.toList());
            String actualMessage = ExceptionUtils.getRootCause(e).getMessage();
            assertTrue(
                    "Actual message is: " + actualMessage,
                    expectedMessages.contains(actualMessage));
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testFitAndPredict() throws Exception {
        StringIndexer stringIndexer =
                new StringIndexer()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
                        .setHandleInvalid(StringIndexerParams.KEEP_INVALID);
        Table output = stringIndexer.fit(trainTable).transform(predictTable)[0];

        List<Row> predictedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedAlphabeticAscPredictData, predictedResult);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testSaveLoadAndPredict() throws Exception {
        StringIndexer stringIndexer =
                new StringIndexer()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
                        .setHandleInvalid(StringIndexerParams.KEEP_INVALID);
        stringIndexer =
                TestUtils.saveAndReload(
                        tEnv, stringIndexer, tempFolder.newFolder().getAbsolutePath());

        StringIndexerModel model = stringIndexer.fit(trainTable);
        model = TestUtils.saveAndReload(tEnv, model, tempFolder.newFolder().getAbsolutePath());

        assertEquals(
                Collections.singletonList("stringArrays"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        Table output = model.transform(predictTable)[0];
        List<Row> predictedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedAlphabeticAscPredictData, predictedResult);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        StringIndexerModel model =
                new StringIndexer()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
                        .fit(trainTable);
        Table modelDataTable = model.getModelData()[0];

        assertEquals(
                Collections.singletonList("stringArrays"),
                modelDataTable.getResolvedSchema().getColumnNames());

        List<StringIndexerModelData> collectedModelData =
                (List<StringIndexerModelData>)
                        (IteratorUtils.toList(
                                StringIndexerModelData.getModelDataStream(modelDataTable)
                                        .executeAndCollect()));
        assertEquals(1, collectedModelData.size());

        StringIndexerModelData modelData = collectedModelData.get(0);
        assertEquals(2, modelData.stringArrays.length);
        assertArrayEquals(expectedAlphabeticAscModelData[0], modelData.stringArrays[0]);
        assertArrayEquals(expectedAlphabeticAscModelData[1], modelData.stringArrays[1]);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testSetModelData() throws Exception {
        StringIndexerModel model =
                new StringIndexer()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setStringOrderType(StringIndexerParams.ALPHABET_ASC_ORDER)
                        .setHandleInvalid(StringIndexerParams.KEEP_INVALID)
                        .fit(trainTable);

        StringIndexerModel newModel = new StringIndexerModel();
        ReadWriteUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(predictTable)[0];

        List<Row> predictedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        verifyPredictionResult(expectedAlphabeticAscPredictData, predictedResult);
    }

    static void verifyPredictionResult(List<Row> expected, List<Row> result) {
        compareResultCollections(
                expected,
                result,
                (row1, row2) -> {
                    int arity = Math.min(row1.getArity(), row2.getArity());
                    for (int i = 0; i < arity; i++) {
                        int cmp =
                                String.valueOf(row1.getField(i))
                                        .compareTo(String.valueOf(row2.getField(i)));
                        if (cmp != 0) {
                            return cmp;
                        }
                    }
                    return 0;
                });
    }
}
