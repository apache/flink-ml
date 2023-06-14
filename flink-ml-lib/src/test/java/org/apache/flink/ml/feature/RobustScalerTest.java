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
import org.apache.flink.ml.feature.robustscaler.RobustScaler;
import org.apache.flink.ml.feature.robustscaler.RobustScalerModel;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Expressions;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** Tests {@link RobustScaler} and {@link RobustScalerModel}. */
public class RobustScalerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table trainDataTable;
    private Table predictDataTable;

    private static final List<Row> TRAIN_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(0, Vectors.dense(0.0, 0.0)),
                            Row.of(1, Vectors.dense(1.0, -1.0)),
                            Row.of(2, Vectors.dense(2.0, -2.0)),
                            Row.of(3, Vectors.dense(3.0, -3.0)),
                            Row.of(4, Vectors.dense(4.0, -4.0)),
                            Row.of(5, Vectors.dense(5.0, -5.0)),
                            Row.of(6, Vectors.dense(6.0, -6.0)),
                            Row.of(7, Vectors.dense(7.0, -7.0)),
                            Row.of(8, Vectors.dense(8.0, -8.0))));
    private static final List<Row> PREDICT_DATA =
            new ArrayList<>(
                    Arrays.asList(
                            Row.of(Vectors.dense(3.0, -3.0)),
                            Row.of(Vectors.dense(6.0, -6.0)),
                            Row.of(Vectors.dense(99.0, -99.0))));
    private static final double EPS = 1.0e-5;

    private static final List<DenseIntDoubleVector> EXPECTED_OUTPUT =
            new ArrayList<>(
                    Arrays.asList(
                            Vectors.dense(0.75, -0.75),
                            Vectors.dense(1.5, -1.5),
                            Vectors.dense(24.75, -24.75)));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        trainDataTable = tEnv.fromDataStream(env.fromCollection(TRAIN_DATA)).as("id", "input");
        predictDataTable = tEnv.fromDataStream(env.fromCollection(PREDICT_DATA)).as("input");
    }

    private static void verifyPredictionResult(
            Table output, String outputCol, List<DenseIntDoubleVector> expected) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<DenseIntDoubleVector> stream =
                tEnv.toDataStream(output)
                        .map(
                                (MapFunction<Row, DenseIntDoubleVector>)
                                        row -> (DenseIntDoubleVector) row.getField(outputCol));
        List<DenseIntDoubleVector> result = IteratorUtils.toList(stream.executeAndCollect());
        TestBaseUtils.compareResultCollections(expected, result, TestUtils::compare);
    }

    @Test
    public void testParam() {
        RobustScaler robustScaler = new RobustScaler();
        assertEquals("input", robustScaler.getInputCol());
        assertEquals("output", robustScaler.getOutputCol());
        assertEquals(0.25, robustScaler.getLower(), EPS);
        assertEquals(0.75, robustScaler.getUpper(), EPS);
        assertEquals(0.001, robustScaler.getRelativeError(), EPS);
        assertFalse(robustScaler.getWithCentering());
        assertTrue(robustScaler.getWithScaling());

        robustScaler
                .setInputCol("test_input")
                .setOutputCol("test_output")
                .setLower(0.1)
                .setUpper(0.9)
                .setRelativeError(0.01)
                .setWithCentering(true)
                .setWithScaling(false);
        assertEquals("test_input", robustScaler.getInputCol());
        assertEquals("test_output", robustScaler.getOutputCol());
        assertEquals(0.1, robustScaler.getLower(), EPS);
        assertEquals(0.9, robustScaler.getUpper(), EPS);
        assertEquals(0.01, robustScaler.getRelativeError(), EPS);
        assertTrue(robustScaler.getWithCentering());
        assertFalse(robustScaler.getWithScaling());
    }

    @Test
    public void testOutputSchema() {
        Table tempTable = trainDataTable.as("id", "test_input");
        RobustScaler robustScaler =
                new RobustScaler().setInputCol("test_input").setOutputCol("test_output");
        RobustScalerModel model = robustScaler.fit(tempTable);
        Table output = model.transform(tempTable)[0];
        assertEquals(
                Arrays.asList("id", "test_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredict() throws Exception {
        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel model = robustScaler.fit(trainDataTable);
        Table output = model.transform(predictDataTable)[0];
        verifyPredictionResult(output, robustScaler.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        trainDataTable =
                TestUtils.convertDataTypesToSparseInt(
                        tEnv, trainDataTable.select(Expressions.$("input")));
        predictDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, predictDataTable);
        assertArrayEquals(
                new Class<?>[] {SparseIntDoubleVector.class},
                TestUtils.getColumnDataTypes(trainDataTable));
        assertArrayEquals(
                new Class<?>[] {SparseIntDoubleVector.class},
                TestUtils.getColumnDataTypes(predictDataTable));

        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel model = robustScaler.fit(trainDataTable);
        Table output = model.transform(predictDataTable)[0];
        verifyPredictionResult(output, robustScaler.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        RobustScaler robustScaler = new RobustScaler();
        RobustScaler loadedRobustScaler =
                TestUtils.saveAndReload(
                        tEnv,
                        robustScaler,
                        tempFolder.newFolder().getAbsolutePath(),
                        RobustScaler::load);
        RobustScalerModel model = loadedRobustScaler.fit(trainDataTable);
        RobustScalerModel loadedModel =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        RobustScalerModel::load);
        assertEquals(
                Arrays.asList("medians", "ranges"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());
        Table output = loadedModel.transform(predictDataTable)[0];
        verifyPredictionResult(output, robustScaler.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testFitOnEmptyData() {
        Table emptyTable =
                tEnv.fromDataStream(env.fromCollection(TRAIN_DATA).filter(x -> x.getArity() == 0))
                        .as("id", "input");
        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel model = robustScaler.fit(emptyTable);
        Table modelDataTable = model.getModelData()[0];
        try {
            modelDataTable.execute().print();
            fail();
        } catch (Throwable e) {
            assertEquals("The training set is empty.", ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testWithCentering() throws Exception {
        RobustScaler robustScaler = new RobustScaler().setWithCentering(true);
        RobustScalerModel model = robustScaler.fit(trainDataTable);
        Table output = model.transform(predictDataTable)[0];
        List<DenseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.dense(-0.25, 0.25),
                                Vectors.dense(0.5, -0.5),
                                Vectors.dense(23.75, -23.75)));
        verifyPredictionResult(output, robustScaler.getOutputCol(), expectedOutput);
    }

    @Test
    public void testWithoutScaling() throws Exception {
        RobustScaler robustScaler = new RobustScaler().setWithCentering(true).setWithScaling(false);
        RobustScalerModel model = robustScaler.fit(trainDataTable);
        Table output = model.transform(predictDataTable)[0];
        List<DenseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.dense(-1, 1),
                                Vectors.dense(2, -2),
                                Vectors.dense(95, -95)));
        verifyPredictionResult(output, robustScaler.getOutputCol(), expectedOutput);
    }

    @Test
    public void testIncompatibleNumOfFeatures() {
        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel model = robustScaler.fit(trainDataTable);

        List<Row> predictData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(Vectors.dense(1.0, 2.0, 3.0)),
                                Row.of(Vectors.dense(-1.0, -2.0, -3.0))));
        Table predictTable = tEnv.fromDataStream(env.fromCollection(predictData)).as("input");
        Table output = model.transform(predictTable)[0];
        try {
            output.execute().print();
            fail();
        } catch (Throwable e) {
            assertTrue(
                    ExceptionUtils.getRootCause(e)
                            .getMessage()
                            .contains("Number of features must be"));
        }
    }

    @Test
    public void testZeroRange() throws Exception {
        List<Row> trainData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(0, Vectors.dense(0.0, 0.0)),
                                Row.of(1, Vectors.dense(1.0, 1.0)),
                                Row.of(2, Vectors.dense(1.0, 1.0)),
                                Row.of(3, Vectors.dense(1.0, 1.0)),
                                Row.of(4, Vectors.dense(4.0, 4.0))));
        List<DenseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.dense(0.0, -0.0),
                                Vectors.dense(0.0, -0.0),
                                Vectors.dense(0.0, -0.0)));
        Table trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("id", "input");
        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel model = robustScaler.fit(trainTable);
        Table output = model.transform(predictDataTable)[0];
        verifyPredictionResult(output, robustScaler.getOutputCol(), expectedOutput);
    }

    @Test
    public void testNaNData() throws Exception {
        List<Row> trainData =
                new ArrayList<>(
                        Arrays.asList(
                                Row.of(0, Vectors.dense(0.0, Double.NaN)),
                                Row.of(1, Vectors.dense(Double.NaN, 0.0)),
                                Row.of(2, Vectors.dense(1.0, -1.0)),
                                Row.of(3, Vectors.dense(2.0, -2.0)),
                                Row.of(4, Vectors.dense(3.0, -3.0)),
                                Row.of(5, Vectors.dense(4.0, -4.0))));
        List<DenseIntDoubleVector> expectedOutput =
                new ArrayList<>(
                        Arrays.asList(
                                Vectors.dense(0.0, Double.NaN),
                                Vectors.dense(Double.NaN, 0.0),
                                Vectors.dense(0.5, -0.5),
                                Vectors.dense(1.0, -1.0),
                                Vectors.dense(1.5, -1.5),
                                Vectors.dense(2.0, -2.0)));
        Table trainTable = tEnv.fromDataStream(env.fromCollection(trainData)).as("id", "input");
        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel model = robustScaler.fit(trainTable);
        Table output = model.transform(trainTable)[0];
        verifyPredictionResult(output, robustScaler.getOutputCol(), expectedOutput);
    }

    @Test
    public void testGetModelData() throws Exception {
        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel model = robustScaler.fit(trainDataTable);
        Table modelData = model.getModelData()[0];
        assertEquals(
                Arrays.asList("medians", "ranges"), modelData.getResolvedSchema().getColumnNames());
        DataStream<Row> output = tEnv.toDataStream(modelData);
        List<Row> modelRows = IteratorUtils.toList(output.executeAndCollect());
        DenseIntDoubleVector medians = (DenseIntDoubleVector) modelRows.get(0).getField(0);
        DenseIntDoubleVector ranges = (DenseIntDoubleVector) modelRows.get(0).getField(1);

        DenseIntDoubleVector expectedMedians = Vectors.dense(4.0, -4.0);
        DenseIntDoubleVector expectedRanges = Vectors.dense(4.0, 4.0);
        assertEquals(expectedMedians, medians);
        assertEquals(expectedRanges, ranges);
    }

    @Test
    public void testSetModelData() throws Exception {
        RobustScaler robustScaler = new RobustScaler();
        RobustScalerModel modelA = robustScaler.fit(trainDataTable);

        Table modelData = modelA.getModelData()[0];
        RobustScalerModel modelB = new RobustScalerModel().setModelData(modelData);
        Table output = modelB.transform(predictDataTable)[0];
        verifyPredictionResult(output, robustScaler.getOutputCol(), EXPECTED_OUTPUT);
    }
}
