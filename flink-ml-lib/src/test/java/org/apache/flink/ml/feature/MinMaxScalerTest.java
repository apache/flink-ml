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
import org.apache.flink.ml.feature.minmaxscaler.MinMaxScaler;
import org.apache.flink.ml.feature.minmaxscaler.MinMaxScalerModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link MinMaxScaler} and {@link MinMaxScalerModel}. */
public class MinMaxScalerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainDataTable;
    private Table predictDataTable;
    private static final List<Row> TRAIN_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(0.0, 3.0)),
                            Row.of(Vectors.dense(2.1, 0.0)),
                            Row.of(Vectors.dense(4.1, 5.1)),
                            Row.of(Vectors.dense(6.1, 8.1)),
                            Row.of(Vectors.dense(200, 400))));
    private static final List<Row> PREDICT_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(150.0, 90.0)),
                            Row.of(Vectors.dense(50.0, 40.0)),
                            Row.of(Vectors.dense(100.0, 50.0))));
    private static final double EPS = 1.0e-5;
    private static final List<DenseVector> EXPECTED_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Vectors.dense(0.25, 0.1),
                            Vectors.dense(0.5, 0.125),
                            Vectors.dense(0.75, 0.225)));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        trainDataTable = tEnv.fromDataStream(env.fromCollection(TRAIN_DATA)).as("input");
        predictDataTable = tEnv.fromDataStream(env.fromCollection(PREDICT_DATA)).as("input");
    }

    private static void verifyPredictionResult(
            Table output, String outputCol, List<DenseVector> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<DenseVector> stream =
                tEnv.toDataStream(output)
                        .map(
                                (MapFunction<Row, DenseVector>)
                                        row -> (DenseVector) row.getField(outputCol));
        List<DenseVector> result = IteratorUtils.toList(stream.executeAndCollect());
        compareResultCollections(expected, result, TestUtils::compare);
    }

    @Test
    public void testParam() {
        MinMaxScaler minMaxScaler = new MinMaxScaler();
        assertEquals("input", minMaxScaler.getInputCol());
        assertEquals("output", minMaxScaler.getOutputCol());
        assertEquals(0.0, minMaxScaler.getMin(), EPS);
        assertEquals(1.0, minMaxScaler.getMax(), EPS);
        minMaxScaler.setInputCol("test_input").setOutputCol("test_output").setMin(1.0).setMax(4.0);
        assertEquals("test_input", minMaxScaler.getInputCol());
        assertEquals(1.0, minMaxScaler.getMin(), EPS);
        assertEquals(4.0, minMaxScaler.getMax(), EPS);
        assertEquals("test_output", minMaxScaler.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        MinMaxScaler minMaxScaler =
                new MinMaxScaler()
                        .setInputCol("test_input")
                        .setOutputCol("test_output")
                        .setMin(1.0)
                        .setMax(4.0);

        MinMaxScalerModel model = minMaxScaler.fit(trainDataTable.as("test_input"));
        Table output = model.transform(predictDataTable.as("test_input"))[0];
        assertEquals(
                Arrays.asList("test_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testMaxValueEqualsMinValueButPredictValueNotEquals() throws Exception {
        List<Row> trainData =
                new ArrayList<>(Collections.singletonList(Row.of(Vectors.dense(40.0, 80.0))));
        Table trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("input");
        List<Row> predictData =
                new ArrayList<>(Collections.singletonList(Row.of(Vectors.dense(30.0, 50.0))));
        Table predictDataTable = tEnv.fromDataStream(env.fromCollection(predictData)).as("input");
        MinMaxScaler minMaxScaler = new MinMaxScaler().setMax(10.0).setMin(0.0);
        MinMaxScalerModel model = minMaxScaler.fit(trainTable);
        Table result = model.transform(predictDataTable)[0];
        verifyPredictionResult(
                result,
                minMaxScaler.getOutputCol(),
                Collections.singletonList(Vectors.dense(5.0, 5.0)));
    }

    @Test
    public void testFitAndPredict() throws Exception {
        MinMaxScaler minMaxScaler = new MinMaxScaler();
        MinMaxScalerModel minMaxScalerModel = minMaxScaler.fit(trainDataTable);
        Table output = minMaxScalerModel.transform(predictDataTable)[0];
        verifyPredictionResult(output, minMaxScaler.getOutputCol(), EXPECTED_DATA);
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        trainDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, trainDataTable);
        predictDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, predictDataTable);
        assertArrayEquals(
                new Class<?>[] {SparseVector.class}, TestUtils.getColumnDataTypes(trainDataTable));
        assertArrayEquals(
                new Class<?>[] {SparseVector.class},
                TestUtils.getColumnDataTypes(predictDataTable));

        MinMaxScaler minMaxScaler = new MinMaxScaler();
        MinMaxScalerModel minMaxScalerModel = minMaxScaler.fit(trainDataTable);
        Table output = minMaxScalerModel.transform(predictDataTable)[0];
        verifyPredictionResult(output, minMaxScaler.getOutputCol(), EXPECTED_DATA);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        MinMaxScaler minMaxScaler = new MinMaxScaler();
        MinMaxScaler loadedMinMaxScaler =
                TestUtils.saveAndReload(
                        tEnv,
                        minMaxScaler,
                        tempFolder.newFolder().getAbsolutePath(),
                        MinMaxScaler::load);
        MinMaxScalerModel model = loadedMinMaxScaler.fit(trainDataTable);
        MinMaxScalerModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        MinMaxScalerModel::load);
        assertEquals(
                Arrays.asList("minVector", "maxVector"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = loadedModel.transform(predictDataTable)[0];
        verifyPredictionResult(output, minMaxScaler.getOutputCol(), EXPECTED_DATA);
    }

    @Test
    public void testGetModelData() throws Exception {
        MinMaxScaler minMaxScaler = new MinMaxScaler();
        MinMaxScalerModel minMaxScalerModel = minMaxScaler.fit(trainDataTable);
        Table modelData = minMaxScalerModel.getModelData()[0];
        assertEquals(
                Arrays.asList("minVector", "maxVector"),
                modelData.getResolvedSchema().getColumnNames());
        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        assertEquals(new DenseVector(new double[] {0.0, 0.0}), modelRows.get(0).getField(0));
        assertEquals(new DenseVector(new double[] {200.0, 400.0}), modelRows.get(0).getField(1));
    }

    @Test
    public void testSetModelData() throws Exception {
        MinMaxScaler minMaxScaler = new MinMaxScaler();
        MinMaxScalerModel modelA = minMaxScaler.fit(trainDataTable);
        Table modelData = modelA.getModelData()[0];
        MinMaxScalerModel modelB = new MinMaxScalerModel().setModelData(modelData);
        ParamUtils.updateExistingParams(modelB, modelA.getParamMap());
        Table output = modelB.transform(predictDataTable)[0];
        verifyPredictionResult(output, minMaxScaler.getOutputCol(), EXPECTED_DATA);
    }
}
