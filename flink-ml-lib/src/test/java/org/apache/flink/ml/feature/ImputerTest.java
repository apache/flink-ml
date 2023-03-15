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

import org.apache.flink.ml.feature.imputer.Imputer;
import org.apache.flink.ml.feature.imputer.ImputerModel;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.apache.flink.ml.feature.imputer.ImputerParams.MEAN;
import static org.apache.flink.ml.feature.imputer.ImputerParams.MEDIAN;
import static org.apache.flink.ml.feature.imputer.ImputerParams.MOST_FREQUENT;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link Imputer} and {@link ImputerModel}. */
public class ImputerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainDataTable;
    private Table predictDataTable;

    private static final double EPS = 1.0e-5;
    private static final List<Row> TRAIN_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Double.NaN, 9.0, 1, 9.0f),
                            Row.of(1.0, 9.0, null, 9.0f),
                            Row.of(1.5, 7.0, 1, 7.0f),
                            Row.of(1.5, Double.NaN, 2, Float.NaN),
                            Row.of(4.0, 5.0, 4, 5.0f),
                            Row.of(null, 4.0, null, 4.0f)));

    private static final List<Row> EXPECTED_MEAN_STRATEGY_OUTPUT =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(2.0, 9.0, 1.0, 9.0),
                            Row.of(1.0, 9.0, 2.0, 9.0),
                            Row.of(1.5, 7.0, 1.0, 7.0),
                            Row.of(1.5, 6.8, 2.0, 6.8),
                            Row.of(4.0, 5.0, 4.0, 5.0),
                            Row.of(2.0, 4.0, 2.0, 4.0)));

    private static final List<Row> EXPECTED_MEDIAN_STRATEGY_OUTPUT =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(1.5, 9.0, 1.0, 9.0),
                            Row.of(1.0, 9.0, 1.0, 9.0),
                            Row.of(1.5, 7.0, 1.0, 7.0),
                            Row.of(1.5, 7.0, 2.0, 7.0),
                            Row.of(4.0, 5.0, 4.0, 5.0),
                            Row.of(1.5, 4.0, 1.0, 4.0)));

    private static final List<Row> EXPECTED_MOST_FREQUENT_STRATEGY_OUTPUT =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(1.5, 9.0, 1.0, 9.0),
                            Row.of(1.0, 9.0, 1.0, 9.0),
                            Row.of(1.5, 7.0, 1.0, 7.0),
                            Row.of(1.5, 9.0, 2.0, 9.0),
                            Row.of(4.0, 5.0, 4.0, 5.0),
                            Row.of(1.5, 4.0, 1.0, 4.0)));

    private static final Map<String, List<Row>> strategyAndExpectedOutputs =
            new HashMap<String, List<Row>>() {
                {
                    put(MEAN, EXPECTED_MEAN_STRATEGY_OUTPUT);
                    put(MEDIAN, EXPECTED_MEDIAN_STRATEGY_OUTPUT);
                    put(MOST_FREQUENT, EXPECTED_MOST_FREQUENT_STRATEGY_OUTPUT);
                }
            };

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        trainDataTable =
                tEnv.fromDataStream(env.fromCollection(TRAIN_DATA)).as("f1", "f2", "f3", "f4");
        predictDataTable =
                tEnv.fromDataStream(env.fromCollection(TRAIN_DATA)).as("f1", "f2", "f3", "f4");
    }

    @SuppressWarnings("unchecked")
    private static void verifyPredictionResult(
            Table output, List<String> outputCols, List<Row> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Row> outputDataStream = tEnv.toDataStream(output);
        List<Row> result = IteratorUtils.toList(outputDataStream.executeAndCollect());
        result =
                result.stream()
                        .map(
                                row -> {
                                    Row outputRow = new Row(outputCols.size());
                                    for (int i = 0; i < outputCols.size(); i++) {
                                        outputRow.setField(i, row.getField(outputCols.get(i)));
                                    }
                                    return outputRow;
                                })
                        .collect(Collectors.toList());
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

    @Test
    public void testParam() {
        Imputer imputer =
                new Imputer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4");
        assertArrayEquals(new String[] {"f1", "f2", "f3", "f4"}, imputer.getInputCols());
        assertArrayEquals(new String[] {"o1", "o2", "o3", "o4"}, imputer.getOutputCols());
        assertEquals(MEAN, imputer.getStrategy());
        assertEquals(Double.NaN, imputer.getMissingValue(), EPS);
        assertEquals(0.001, imputer.getRelativeError(), EPS);

        imputer.setMissingValue(0.0)
                .setStrategy(MEDIAN)
                .setRelativeError(0.1)
                .setInputCols("f1", "f2")
                .setOutputCols("o1", "o2");
        assertEquals(MEDIAN, imputer.getStrategy());
        assertEquals(0.0, imputer.getMissingValue(), EPS);
        assertEquals(0.1, imputer.getRelativeError(), EPS);
        assertArrayEquals(new String[] {"f1", "f2"}, imputer.getInputCols());
        assertArrayEquals(new String[] {"o1", "o2"}, imputer.getOutputCols());
    }

    @Test
    public void testOutputSchema() {
        Imputer imputer =
                new Imputer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4");
        ImputerModel model = imputer.fit(trainDataTable);
        Table output = model.transform(predictDataTable)[0];
        assertEquals(
                Arrays.asList("f1", "f2", "f3", "f4", "o1", "o2", "o3", "o4"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        for (Map.Entry<String, List<Row>> entry : strategyAndExpectedOutputs.entrySet()) {
            Imputer imputer =
                    new Imputer()
                            .setInputCols("f1", "f2", "f3", "f4")
                            .setOutputCols("o1", "o2", "o3", "o4")
                            .setStrategy(entry.getKey());
            ImputerModel model = imputer.fit(trainDataTable);
            Table output = model.transform(predictDataTable)[0];
            verifyPredictionResult(output, Arrays.asList("o1", "o2", "o3", "o4"), entry.getValue());
        }
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        Imputer imputer =
                new Imputer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4");
        Imputer loadedImputer =
                TestUtils.saveAndReload(
                        tEnv, imputer, tempFolder.newFolder().getAbsolutePath(), Imputer::load);
        ImputerModel model = loadedImputer.fit(trainDataTable);
        ImputerModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv, model, tempFolder.newFolder().getAbsolutePath(), ImputerModel::load);
        assertEquals(
                Collections.singletonList("surrogates"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = loadedModel.transform(predictDataTable)[0];
        verifyPredictionResult(
                output, Arrays.asList(imputer.getOutputCols()), EXPECTED_MEAN_STRATEGY_OUTPUT);
    }

    @Test
    public void testFitOnEmptyData() {
        Table emptyTable =
                tEnv.fromDataStream(env.fromCollection(TRAIN_DATA).filter(x -> x.getArity() == 0))
                        .as("f1", "f2", "f3", "f4");

        strategyAndExpectedOutputs.remove(MEDIAN);
        for (Map.Entry<String, List<Row>> entry : strategyAndExpectedOutputs.entrySet()) {
            Imputer imputer =
                    new Imputer()
                            .setInputCols("f1", "f2", "f3", "f4")
                            .setOutputCols("o1", "o2", "o3", "o4")
                            .setStrategy(entry.getKey());
            ImputerModel model = imputer.fit(emptyTable);
            Table modelDataTable = model.getModelData()[0];
            try {
                modelDataTable.execute().print();
                fail();
            } catch (Throwable e) {
                assertEquals(
                        "The training set is empty or does not contains valid data.",
                        ExceptionUtils.getRootCause(e).getMessage());
            }
        }
    }

    @Test
    public void testNoValidDataOnMedianStrategy() {
        final List<Row> trainData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(Double.NaN, 3.0f), Row.of(null, 2.0f), Row.of(1.0, 1.0f)));
        trainDataTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("f1", "f2");
        Imputer imputer =
                new Imputer()
                        .setInputCols("f1", "f2")
                        .setOutputCols("o1", "o2")
                        .setStrategy(MEDIAN)
                        .setMissingValue(1.0);
        ImputerModel model = imputer.fit(trainDataTable);
        Table modelDataTable = model.getModelData()[0];
        try {
            modelDataTable.execute().print();
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "Surrogate cannot be computed. All the values in column [f1] are null, NaN or missingValue.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testMultipleModeOnMostFrequentStrategy() throws Exception {
        final List<Row> trainData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(1.0, 2.0),
                                Row.of(1.0, 2.0),
                                Row.of(2.0, 1.0),
                                Row.of(2.0, 1.0)));
        trainDataTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("f1", "f2");
        Imputer imputer =
                new Imputer()
                        .setInputCols("f1", "f2")
                        .setOutputCols("o1", "o2")
                        .setStrategy(MOST_FREQUENT);
        ImputerModel model = imputer.fit(trainDataTable);
        Table modelData = model.getModelData()[0];
        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        Map<String, Double> surrogates = (Map<String, Double>) modelRows.get(0).getField(0);
        assert surrogates != null;
        assertEquals(1.0, surrogates.get("f1"), EPS);
        assertEquals(1.0, surrogates.get("f2"), EPS);
    }

    @Test
    public void testInconsistentInputsAndOutputs() {
        Imputer imputer =
                new Imputer().setInputCols("f1", "f2", "f3", "f4").setOutputCols("o1", "o2", "o3");
        try {
            imputer.fit(trainDataTable);
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "Num of input columns and output columns are inconsistent.", e.getMessage());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        Imputer imputer =
                new Imputer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4");
        ImputerModel model = imputer.fit(trainDataTable);
        Table modelData = model.getModelData()[0];
        assertEquals(
                Collections.singletonList("surrogates"),
                modelData.getResolvedSchema().getColumnNames());
        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        Map<String, Double> surrogates = (Map<String, Double>) modelRows.get(0).getField(0);
        assert surrogates != null;
        assertEquals(2.0, surrogates.get("f1"), EPS);
        assertEquals(6.8, surrogates.get("f2"), EPS);
        assertEquals(2.0, surrogates.get("f3"), EPS);
        assertEquals(6.8, surrogates.get("f4"), EPS);
    }

    @Test
    public void testSetModelData() throws Exception {
        Imputer imputer =
                new Imputer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4");
        ImputerModel modelA = imputer.fit(trainDataTable);

        Table modelData = modelA.getModelData()[0];
        ImputerModel modelB =
                new ImputerModel()
                        .setModelData(modelData)
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4");
        Table output = modelB.transform(predictDataTable)[0];
        verifyPredictionResult(
                output, Arrays.asList(imputer.getOutputCols()), EXPECTED_MEAN_STRATEGY_OUTPUT);
    }
}
