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

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.vectorindexer.VectorIndexer;
import org.apache.flink.ml.feature.vectorindexer.VectorIndexerModel;
import org.apache.flink.ml.feature.vectorindexer.VectorIndexerModelData;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Expressions;
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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link VectorIndexer} and {@link VectorIndexerModel}. */
public class VectorIndexerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainInputTable;
    private Table testInputTable;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        List<Row> trainInput =
                Arrays.asList(
                        Row.of(Vectors.dense(1, 1)),
                        Row.of(Vectors.dense(2, -1)),
                        Row.of(Vectors.dense(3, 1)),
                        Row.of(Vectors.dense(4, 0)),
                        Row.of(Vectors.dense(5, 0)));
        List<Row> testInput =
                Arrays.asList(
                        Row.of(Vectors.dense(0, 2)),
                        Row.of(Vectors.dense(0, 0)),
                        Row.of(Vectors.dense(0, -1)));
        trainInputTable = tEnv.fromDataStream(env.fromCollection(trainInput)).as("input");
        testInputTable = tEnv.fromDataStream(env.fromCollection(testInput)).as("input");
    }

    @Test
    public void testParam() {
        VectorIndexer vectorIndexer = new VectorIndexer();
        assertEquals("input", vectorIndexer.getInputCol());
        assertEquals("output", vectorIndexer.getOutputCol());
        assertEquals(20, vectorIndexer.getMaxCategories());
        assertEquals(HasHandleInvalid.ERROR_INVALID, vectorIndexer.getHandleInvalid());

        vectorIndexer
                .setInputCol("test_input")
                .setOutputCol("test_output")
                .setMaxCategories(3)
                .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);

        assertEquals("test_input", vectorIndexer.getInputCol());
        assertEquals("test_output", vectorIndexer.getOutputCol());
        assertEquals(3, vectorIndexer.getMaxCategories());
        assertEquals(HasHandleInvalid.KEEP_INVALID, vectorIndexer.getHandleInvalid());
    }

    @Test
    public void testOutputSchema() {
        VectorIndexer vectorIndexer = new VectorIndexer();
        Table output = vectorIndexer.fit(trainInputTable).transform(trainInputTable)[0];

        assertEquals(
                Arrays.asList(vectorIndexer.getInputCol(), vectorIndexer.getOutputCol()),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredictOnSparseInput() throws Exception {
        List<Row> sparseTrainInput =
                Arrays.asList(
                        Row.of(Vectors.sparse(2, new int[] {0}, new double[] {1})),
                        Row.of(Vectors.sparse(2, new int[] {0, 1}, new double[] {2, -1})),
                        Row.of(Vectors.sparse(2, new int[] {0, 1}, new double[] {3, 1})),
                        Row.of(Vectors.sparse(2, new int[] {0}, new double[] {4})),
                        Row.of(Vectors.sparse(2, new int[] {0}, new double[] {5})));

        List<Row> sparseTestInput =
                Collections.singletonList(
                        Row.of(Vectors.sparse(2, new int[] {0, 1}, new double[] {0, 2})));
        Table sparseTrainTable =
                tEnv.fromDataStream(env.fromCollection(sparseTrainInput)).as("input");
        Table sparseTestTable =
                tEnv.fromDataStream(env.fromCollection(sparseTestInput)).as("input");

        Table output =
                new VectorIndexer()
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID)
                        .setMaxCategories(3)
                        .fit(sparseTrainTable)
                        .transform(sparseTestTable)[0];

        List<Row> expectedOutput =
                Collections.singletonList(
                        Row.of(Vectors.sparse(2, new int[] {0, 1}, new double[] {0, 3})));
        verifyPredictionResult(expectedOutput, output, "output");
    }

    @Test
    public void testFitAndPredictWithLargeMaxCategories() throws Exception {
        VectorIndexer vectorIndexer =
                new VectorIndexer()
                        .setMaxCategories(Integer.MAX_VALUE)
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID);

        Table output = vectorIndexer.fit(trainInputTable).transform(testInputTable)[0];
        List<Row> expectedOutput =
                Arrays.asList(
                        Row.of(Vectors.dense(5, 3)),
                        Row.of(Vectors.dense(5, 0)),
                        Row.of(Vectors.dense(5, 1)));
        verifyPredictionResult(expectedOutput, output, vectorIndexer.getOutputCol());
    }

    @Test
    public void testFitAndPredictWithHandleInvalid() throws Exception {
        Table output;
        List<Row> expectedOutput;
        VectorIndexer vectorIndexer = new VectorIndexer().setMaxCategories(3);

        // Keeps invalid data.
        expectedOutput =
                Arrays.asList(
                        Row.of(Vectors.dense(0, 3)),
                        Row.of(Vectors.dense(0, 0)),
                        Row.of(Vectors.dense(0, 1)));
        vectorIndexer.setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        output = vectorIndexer.fit(trainInputTable).transform(testInputTable)[0];
        verifyPredictionResult(expectedOutput, output, vectorIndexer.getOutputCol());

        // Skips invalid data.
        vectorIndexer.setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        expectedOutput = Arrays.asList(Row.of(Vectors.dense(0, 0)), Row.of(Vectors.dense(0, 1)));
        output = vectorIndexer.fit(trainInputTable).transform(testInputTable)[0];
        verifyPredictionResult(expectedOutput, output, vectorIndexer.getOutputCol());

        // Throws an exception on invalid data.
        vectorIndexer.setHandleInvalid(HasHandleInvalid.ERROR_INVALID);
        try {
            output = vectorIndexer.fit(trainInputTable).transform(testInputTable)[0];
            IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "The input contains unseen double: 2.0. "
                            + "See "
                            + HasHandleInvalid.HANDLE_INVALID
                            + " parameter for more options.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        VectorIndexer vectorIndexer =
                new VectorIndexer().setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        vectorIndexer =
                TestUtils.saveAndReload(
                        tEnv,
                        vectorIndexer,
                        tempFolder.newFolder().getAbsolutePath(),
                        VectorIndexer::load);

        VectorIndexerModel model = vectorIndexer.fit(trainInputTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        VectorIndexerModel::load);

        assertEquals(
                Collections.singletonList("categoryMaps"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        Table output = model.transform(testInputTable)[0];
        List<Row> expectedOutput =
                Arrays.asList(
                        Row.of(Vectors.dense(5, 3)),
                        Row.of(Vectors.dense(5, 0)),
                        Row.of(Vectors.dense(5, 1)));
        verifyPredictionResult(expectedOutput, output, vectorIndexer.getOutputCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        VectorIndexer vectorIndexer = new VectorIndexer().setMaxCategories(3);
        VectorIndexerModel model = vectorIndexer.fit(trainInputTable);
        Table modelDataTable = model.getModelData()[0];

        assertEquals(
                Collections.singletonList("categoryMaps"),
                modelDataTable.getResolvedSchema().getColumnNames());

        List<VectorIndexerModelData> collectedModelData =
                (List<VectorIndexerModelData>)
                        IteratorUtils.toList(
                                VectorIndexerModelData.getModelDataStream(modelDataTable)
                                        .executeAndCollect());

        assertEquals(1, collectedModelData.size());
        HashMap<Double, Integer> column1ModelData = new HashMap<>();
        column1ModelData.put(-1.0, 1);
        column1ModelData.put(0.0, 0);
        column1ModelData.put(1.0, 2);
        assertEquals(
                Collections.singletonMap(1, column1ModelData),
                collectedModelData.get(0).categoryMaps);
    }

    @Test
    public void testSetModelData() throws Exception {
        VectorIndexer vectorIndexer =
                new VectorIndexer().setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        VectorIndexerModel model = vectorIndexer.fit(trainInputTable);

        VectorIndexerModel newModel = new VectorIndexerModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(testInputTable)[0];

        List<Row> expectedOutput =
                Arrays.asList(
                        Row.of(Vectors.dense(5, 3)),
                        Row.of(Vectors.dense(5, 0)),
                        Row.of(Vectors.dense(5, 1)));
        verifyPredictionResult(expectedOutput, output, newModel.getOutputCol());
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(List<Row> expectedOutput, Table output, String outputCol)
            throws Exception {
        List<Row> collectedResult =
                IteratorUtils.toList(
                        tEnv.toDataStream(output.select(Expressions.$(outputCol)))
                                .executeAndCollect());
        compareResultCollections(
                expectedOutput,
                collectedResult,
                Comparator.comparingInt(o -> (o.getField(0)).hashCode()));
    }
}
