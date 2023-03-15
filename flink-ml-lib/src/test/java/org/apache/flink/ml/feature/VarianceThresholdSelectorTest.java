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
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.feature.variancethresholdselector.VarianceThresholdSelector;
import org.apache.flink.ml.feature.variancethresholdselector.VarianceThresholdSelectorModel;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
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
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Tests {@link VarianceThresholdSelector} and {@link VarianceThresholdSelectorModel}. */
public class VarianceThresholdSelectorTest extends AbstractTestBase {

    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainDataTable;
    private Table predictDataTable;

    private static final double EPS = 1.0e-5;
    private static final List<Row> TRAIN_DATA =
            Arrays.asList(
                    Row.of(1, Vectors.dense(5.0, 7.0, 0.0, 7.0, 6.0, 0.0)),
                    Row.of(2, Vectors.dense(0.0, 9.0, 6.0, 0.0, 5.0, 9.0).toSparse()),
                    Row.of(3, Vectors.dense(0.0, 9.0, 3.0, 0.0, 5.0, 5.0)),
                    Row.of(4, Vectors.dense(1.0, 9.0, 8.0, 5.0, 7.0, 4.0).toSparse()),
                    Row.of(5, Vectors.dense(9.0, 8.0, 6.0, 5.0, 4.0, 4.0)),
                    Row.of(6, Vectors.dense(6.0, 9.0, 7.0, 0.0, 2.0, 0.0).toSparse()));

    private static final List<Row> PREDICT_DATA =
            Arrays.asList(
                    Row.of(Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
                    Row.of(Vectors.dense(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
                    Row.of(Vectors.sparse(6, new int[] {0, 3, 4}, new double[] {0.1, 0.3, 0.5})));

    private static final List<Vector> EXPECTED_OUTPUT =
            Arrays.asList(
                    Vectors.dense(1.0, 4.0, 6.0),
                    Vectors.dense(0.1, 0.4, 0.6),
                    Vectors.sparse(3, new int[] {0, 1}, new double[] {0.1, 0.3}));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        trainDataTable =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        TRAIN_DATA, Types.ROW(Types.INT, VectorTypeInfo.INSTANCE)))
                        .as("id", "input");
        predictDataTable =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        PREDICT_DATA, Types.ROW(VectorTypeInfo.INSTANCE)))
                        .as("input");
    }

    private static void verifyPredictionResult(
            Table output, String outputCol, List<Vector> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Vector> stream =
                tEnv.toDataStream(output)
                        .map(
                                (MapFunction<Row, Vector>) row -> (Vector) row.getField(outputCol),
                                VectorTypeInfo.INSTANCE);
        List<Vector> result = IteratorUtils.toList(stream.executeAndCollect());
        compareResultCollections(expected, result, TestUtils::compare);
    }

    @Test
    public void testParam() {
        VarianceThresholdSelector varianceThresholdSelector = new VarianceThresholdSelector();
        assertEquals("input", varianceThresholdSelector.getInputCol());
        assertEquals("output", varianceThresholdSelector.getOutputCol());
        assertEquals(0.0, varianceThresholdSelector.getVarianceThreshold(), EPS);

        varianceThresholdSelector
                .setInputCol("test_input")
                .setOutputCol("test_output")
                .setVarianceThreshold(0.5);
        assertEquals("test_input", varianceThresholdSelector.getInputCol());
        assertEquals(0.5, varianceThresholdSelector.getVarianceThreshold(), EPS);
        assertEquals("test_output", varianceThresholdSelector.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainDataTable.as("id", "test_input");
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector()
                        .setInputCol("test_input")
                        .setOutputCol("test_output");
        VarianceThresholdSelectorModel model = varianceThresholdSelector.fit(tempTable);
        Table output = model.transform(tempTable)[0];
        assertEquals(
                Arrays.asList("id", "test_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector().setVarianceThreshold(8.0);
        VarianceThresholdSelectorModel model = varianceThresholdSelector.fit(trainDataTable);
        Table predictTableOutput = model.transform(predictDataTable)[0];
        verifyPredictionResult(
                predictTableOutput, varianceThresholdSelector.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testNonSelectedFeatures() throws Exception {
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector().setVarianceThreshold(100.0);
        VarianceThresholdSelectorModel model = varianceThresholdSelector.fit(trainDataTable);
        Table predictTableOutput = model.transform(predictDataTable)[0];
        verifyPredictionResult(
                predictTableOutput,
                varianceThresholdSelector.getOutputCol(),
                Arrays.asList(Vectors.dense(), Vectors.dense(), Vectors.dense()));
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector().setVarianceThreshold(8.0);
        VarianceThresholdSelector loadedVarianceThresholdSelector =
                TestUtils.saveAndReload(
                        tEnv,
                        varianceThresholdSelector,
                        tempFolder.newFolder().getAbsolutePath(),
                        VarianceThresholdSelector::load);
        VarianceThresholdSelectorModel model = loadedVarianceThresholdSelector.fit(trainDataTable);
        VarianceThresholdSelectorModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        VarianceThresholdSelectorModel::load);
        assertEquals(
                Arrays.asList("numOfFeatures", "indices"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = loadedModel.transform(predictDataTable)[0];
        verifyPredictionResult(output, varianceThresholdSelector.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testFitOnEmptyData() {
        Table emptyTable =
                tEnv.fromDataStream(
                                env.fromCollection(
                                                TRAIN_DATA,
                                                Types.ROW(Types.INT, VectorTypeInfo.INSTANCE))
                                        .filter(x -> x.getArity() == 0))
                        .as("id", "input");

        VarianceThresholdSelector varianceThresholdSelector = new VarianceThresholdSelector();
        VarianceThresholdSelectorModel model = varianceThresholdSelector.fit(emptyTable);
        Table modelDataTable = model.getModelData()[0];
        try {
            modelDataTable.execute().print();
            fail();
        } catch (Throwable e) {
            assertEquals("The training set is empty.", ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testIncompatibleNumOfFeatures() {
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector().setVarianceThreshold(8.0);
        VarianceThresholdSelectorModel model = varianceThresholdSelector.fit(trainDataTable);

        List<Row> predictData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(Vectors.dense(1.0, 2.0, 3.0, 4.0)),
                                Row.of(Vectors.dense(0.1, 0.2, 0.3, 0.4))));
        Table predictTable = tEnv.fromDataStream(env.fromCollection(predictData)).as("input");
        Table output = model.transform(predictTable)[0];
        try {
            output.execute().print();
            fail();
        } catch (Throwable e) {
            assertTrue(
                    ExceptionUtils.getRootCause(e)
                            .getMessage()
                            .contains("but VarianceThresholdSelector is expecting"));
        }
    }

    @Test
    public void testGetModelData() throws Exception {
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector().setVarianceThreshold(8.0);
        VarianceThresholdSelectorModel model = varianceThresholdSelector.fit(trainDataTable);
        Table modelData = model.getModelData()[0];
        assertEquals(
                Arrays.asList("numOfFeatures", "indices"),
                modelData.getResolvedSchema().getColumnNames());
        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        long numOfFeatures = (int) modelRows.get(0).getField(0);
        int[] indices = (int[]) modelRows.get(0).getField(1);
        assertEquals(6, numOfFeatures);
        int[] expectedIndices = {0, 3, 5};
        for (int i = 0; i < indices.length; i++) {
            assertEquals(expectedIndices[i], indices[i]);
        }
    }

    @Test
    public void testSetModelData() throws Exception {
        VarianceThresholdSelector varianceThresholdSelector =
                new VarianceThresholdSelector().setVarianceThreshold(8.0);
        VarianceThresholdSelectorModel modelA = varianceThresholdSelector.fit(trainDataTable);

        Table modelData = modelA.getModelData()[0];
        VarianceThresholdSelectorModel modelB =
                new VarianceThresholdSelectorModel().setModelData(modelData);
        Table output = modelB.transform(predictDataTable)[0];
        verifyPredictionResult(output, varianceThresholdSelector.getOutputCol(), EXPECTED_OUTPUT);
    }
}
