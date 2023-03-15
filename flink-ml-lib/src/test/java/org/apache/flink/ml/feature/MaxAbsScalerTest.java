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
import org.apache.flink.ml.feature.maxabsscaler.MaxAbsScaler;
import org.apache.flink.ml.feature.maxabsscaler.MaxAbsScalerModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
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
import java.util.List;

import static org.apache.flink.test.util.TestBaseUtils.compareResultCollections;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link MaxAbsScaler} and {@link MaxAbsScalerModel}. */
public class MaxAbsScalerTest {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;

    private Table trainDataTable;
    private Table predictDataTable;
    private Table trainSparseDataTable;
    private Table predictSparseDataTable;

    private static final List<Row> TRAIN_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(0.0, 3.0, 0.0)),
                            Row.of(Vectors.dense(2.1, 0.0, 0.0)),
                            Row.of(Vectors.dense(4.1, 5.1, 0.0)),
                            Row.of(Vectors.dense(6.1, 8.1, 0.0)),
                            Row.of(Vectors.dense(200, -400, 0.0))));

    private static final List<Row> PREDICT_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(150.0, 90.0, 1.0)),
                            Row.of(Vectors.dense(50.0, 40.0, 1.0)),
                            Row.of(Vectors.dense(100.0, 50.0, 0.5))));

    private static final List<Row> TRAIN_SPARSE_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.sparse(4, new int[] {1, 3}, new double[] {4.0, 3.0})),
                            Row.of(Vectors.sparse(4, new int[] {0, 2}, new double[] {2.0, -6.0})),
                            Row.of(Vectors.sparse(4, new int[] {1, 2}, new double[] {1.0, 3.0})),
                            Row.of(Vectors.sparse(4, new int[] {0, 1}, new double[] {2.0, 8.0})),
                            Row.of(Vectors.sparse(4, new int[] {1, 3}, new double[] {1.0, 5.0}))));

    private static final List<Row> PREDICT_SPARSE_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.sparse(4, new int[] {0, 1}, new double[] {2.0, 4.0})),
                            Row.of(Vectors.sparse(4, new int[] {0, 2}, new double[] {1.0, 3.0})),
                            Row.of(Vectors.sparse(4, new int[] {}, new double[] {})),
                            Row.of(Vectors.sparse(4, new int[] {1, 3}, new double[] {1.0, 2.0}))));

    private static final List<Vector> EXPECTED_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Vectors.dense(0.25, 0.1, 1.0),
                            Vectors.dense(0.5, 0.125, 0.5),
                            Vectors.dense(0.75, 0.225, 1.0)));

    private static final List<Vector> EXPECTED_SPARSE_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Vectors.sparse(4, new int[] {0, 1}, new double[] {1.0, 0.5}),
                            Vectors.sparse(4, new int[] {0, 2}, new double[] {0.5, 0.5}),
                            Vectors.sparse(4, new int[] {}, new double[] {}),
                            Vectors.sparse(4, new int[] {1, 3}, new double[] {0.125, 0.4})));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        trainDataTable = tEnv.fromDataStream(env.fromCollection(TRAIN_DATA)).as("input");
        predictDataTable = tEnv.fromDataStream(env.fromCollection(PREDICT_DATA)).as("input");

        trainSparseDataTable =
                tEnv.fromDataStream(env.fromCollection(TRAIN_SPARSE_DATA)).as("input");
        predictSparseDataTable =
                tEnv.fromDataStream(env.fromCollection(PREDICT_SPARSE_DATA)).as("input");
    }

    private static void verifyPredictionResult(
            Table output, String outputCol, List<Vector> expectedData) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();

        DataStream<Vector> stream =
                tEnv.toDataStream(output)
                        .map(
                                (MapFunction<Row, Vector>) row -> row.getFieldAs(outputCol),
                                VectorTypeInfo.INSTANCE);

        List<Vector> result = IteratorUtils.toList(stream.executeAndCollect());
        compareResultCollections(expectedData, result, TestUtils::compare);
    }

    @Test
    public void testParam() {
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();
        assertEquals("input", maxAbsScaler.getInputCol());
        assertEquals("output", maxAbsScaler.getOutputCol());

        maxAbsScaler.setInputCol("test_input").setOutputCol("test_output");
        assertEquals("test_input", maxAbsScaler.getInputCol());
        assertEquals("test_output", maxAbsScaler.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        MaxAbsScaler maxAbsScaler =
                new MaxAbsScaler().setInputCol("test_input").setOutputCol("test_output");

        MaxAbsScalerModel model = maxAbsScaler.fit(trainDataTable.as("test_input"));
        Table output = model.transform(predictDataTable.as("test_input"))[0];
        assertEquals(
                Arrays.asList("test_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();
        MaxAbsScalerModel maxAbsScalerModel = maxAbsScaler.fit(trainDataTable);
        Table output = maxAbsScalerModel.transform(predictDataTable)[0];
        verifyPredictionResult(output, maxAbsScaler.getOutputCol(), EXPECTED_DATA);
    }

    @Test
    public void testFitDataWithNullValue() {
        List<Row> trainData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(Vectors.dense(0.0, 3.0)),
                                Row.of(Vectors.dense(2.1, 0.0)),
                                Row.of((Object) null),
                                Row.of(Vectors.dense(6.1, 8.1)),
                                Row.of(Vectors.dense(200, 400))));

        Table trainDataWithInvalidData =
                tEnv.fromDataStream(env.fromCollection(trainData)).as("input");

        try {
            MaxAbsScaler maxAbsScaler = new MaxAbsScaler();
            MaxAbsScalerModel model = maxAbsScaler.fit(trainDataWithInvalidData);
            IteratorUtils.toList(tEnv.toDataStream(model.getModelData()[0]).executeAndCollect());
            fail();
        } catch (Exception e) {
            assertEquals(
                    "The vector must not be null.", ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testFitAndPredictSparse() throws Exception {
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();
        MaxAbsScalerModel maxAbsScalerModel = maxAbsScaler.fit(trainSparseDataTable);
        Table output = maxAbsScalerModel.transform(predictSparseDataTable)[0];
        verifyPredictionResult(output, maxAbsScaler.getOutputCol(), EXPECTED_SPARSE_DATA);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();
        MaxAbsScaler loadedMaxAbsScaler =
                TestUtils.saveAndReload(
                        tEnv,
                        maxAbsScaler,
                        tempFolder.newFolder().getAbsolutePath(),
                        MaxAbsScaler::load);

        MaxAbsScalerModel model = loadedMaxAbsScaler.fit(trainDataTable);
        MaxAbsScalerModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        MaxAbsScalerModel::load);

        Table output = loadedModel.transform(predictDataTable)[0];
        verifyPredictionResult(output, maxAbsScaler.getOutputCol(), EXPECTED_DATA);
    }

    @Test
    public void testGetModelData() throws Exception {
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();
        MaxAbsScalerModel maxAbsScalerModel = maxAbsScaler.fit(trainDataTable);

        Table modelData = maxAbsScalerModel.getModelData()[0];
        assertEquals(
                Collections.singletonList("maxVector"),
                modelData.getResolvedSchema().getColumnNames());

        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        assertEquals(
                new DenseVector(new double[] {200.0, 400.0, 0.0}), modelRows.get(0).getField(0));
    }

    @Test
    public void testSetModelData() throws Exception {
        MaxAbsScaler maxAbsScaler = new MaxAbsScaler();
        MaxAbsScalerModel modelA = maxAbsScaler.fit(trainDataTable);
        Table modelData = modelA.getModelData()[0];

        MaxAbsScalerModel modelB = new MaxAbsScalerModel().setModelData(modelData);
        ParamUtils.updateExistingParams(modelB, modelA.getParamMap());
        Table output = modelB.transform(predictDataTable)[0];
        verifyPredictionResult(output, maxAbsScaler.getOutputCol(), EXPECTED_DATA);
    }
}
