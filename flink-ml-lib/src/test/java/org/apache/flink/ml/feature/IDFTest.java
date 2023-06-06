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

import org.apache.flink.ml.feature.idf.IDF;
import org.apache.flink.ml.feature.idf.IDFModel;
import org.apache.flink.ml.feature.idf.IDFModelData;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.flink.table.api.Expressions.$;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link IDF} and {@link IDFModel}. */
public class IDFTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table inputTable;

    private static final List<DenseIntDoubleVector> expectedOutput =
            Arrays.asList(
                    Vectors.dense(0, 0, 0, 0.5753641),
                    Vectors.dense(0, 0, 1.3862943, 0.8630462),
                    Vectors.dense(0, 0, 0, 0));
    private static final List<DenseIntDoubleVector> expectedOutputMinDocFreqAsTwo =
            Arrays.asList(
                    Vectors.dense(0, 0, 0, 0.5753641),
                    Vectors.dense(0, 0, 0, 0.8630462),
                    Vectors.dense(0, 0, 0, 0));
    private static final double TOLERANCE = 1e-7;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        List<DenseIntDoubleVector> input =
                Arrays.asList(
                        Vectors.dense(0, 1, 0, 2),
                        Vectors.dense(0, 1, 2, 3),
                        Vectors.dense(0, 1, 0, 0));
        inputTable = tEnv.fromDataStream(env.fromCollection(input).map(x -> x)).as("input");
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(
            List<DenseIntDoubleVector> expectedOutput, Table output, String predictionCol)
            throws Exception {
        List<Row> collectedResult =
                IteratorUtils.toList(
                        tEnv.toDataStream(output.select($(predictionCol))).executeAndCollect());
        List<DenseIntDoubleVector> actualOutputs = new ArrayList<>(expectedOutput.size());
        collectedResult.forEach(x -> actualOutputs.add((x.getFieldAs(0))));

        actualOutputs.sort(TestUtils::compare);
        expectedOutput.sort(TestUtils::compare);
        assertEquals(expectedOutput.size(), collectedResult.size());
        for (int i = 0; i < expectedOutput.size(); i++) {
            assertArrayEquals(expectedOutput.get(i).values, actualOutputs.get(i).values, TOLERANCE);
        }
    }

    @Test
    public void testParam() {
        IDF idf = new IDF();
        assertEquals("input", idf.getInputCol());
        assertEquals(0, idf.getMinDocFreq());
        assertEquals("output", idf.getOutputCol());

        idf.setInputCol("test_input").setMinDocFreq(2).setOutputCol("test_output");

        assertEquals("test_input", idf.getInputCol());
        assertEquals(2, idf.getMinDocFreq());
        assertEquals("test_output", idf.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable =
                tEnv.fromDataStream(env.fromElements(Row.of("", "")))
                        .as("test_input", "dummy_input");
        IDF idf = new IDF().setInputCol("test_input").setOutputCol("test_output");
        Table output = idf.fit(tempTable).transform(tempTable)[0];

        assertEquals(
                Arrays.asList("test_input", "dummy_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        IDF idf = new IDF();
        Table output;

        // Tests minDocFreq = 0.
        output = idf.fit(inputTable).transform(inputTable)[0];
        verifyPredictionResult(expectedOutput, output, idf.getOutputCol());

        // Tests minDocFreq = 2.
        idf.setMinDocFreq(2);
        output = idf.fit(inputTable).transform(inputTable)[0];
        verifyPredictionResult(expectedOutputMinDocFreqAsTwo, output, idf.getOutputCol());
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        IDF idf = new IDF();
        idf =
                TestUtils.saveAndReload(
                        tEnv, idf, tempFolder.newFolder().getAbsolutePath(), IDF::load);

        IDFModel model = idf.fit(inputTable);
        model =
                TestUtils.saveAndReload(
                        tEnv, model, tempFolder.newFolder().getAbsolutePath(), IDFModel::load);

        assertEquals(
                Arrays.asList("idf", "docFreq", "numDocs"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        Table output = model.transform(inputTable)[0];
        verifyPredictionResult(expectedOutput, output, idf.getOutputCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        IDFModel model = new IDF().fit(inputTable);
        Table modelDataTable = model.getModelData()[0];

        assertEquals(
                Arrays.asList("idf", "docFreq", "numDocs"),
                modelDataTable.getResolvedSchema().getColumnNames());

        List<IDFModelData> collectedModelData =
                (List<IDFModelData>)
                        IteratorUtils.toList(
                                IDFModelData.getModelDataStream(modelDataTable)
                                        .executeAndCollect());

        assertEquals(1, collectedModelData.size());
        IDFModelData modelData = collectedModelData.get(0);
        assertEquals(3, modelData.numDocs);
        assertArrayEquals(new long[] {0, 3, 1, 2}, modelData.docFreq);
        assertArrayEquals(
                new double[] {1.3862943, 0, 0.6931471, 0.2876820}, modelData.idf.values, TOLERANCE);
    }

    @Test
    public void testSetModelData() throws Exception {
        IDFModel model = new IDF().fit(inputTable);

        IDFModel newModel = new IDFModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(inputTable)[0];

        verifyPredictionResult(expectedOutput, output, model.getOutputCol());
    }

    @Test
    public void testFitOnEmptyData() {
        Table emptyTable =
                tEnv.fromDataStream(env.fromElements(Row.of(1, 2)).filter(x -> x.getArity() == 0))
                        .as("input");
        IDFModel model = new IDF().fit(emptyTable);
        Table modelDataTable = model.getModelData()[0];

        try {
            modelDataTable.execute().collect().next();
            fail();
        } catch (Throwable e) {
            assertEquals("The training set is empty.", ExceptionUtils.getRootCause(e).getMessage());
        }
    }
}
