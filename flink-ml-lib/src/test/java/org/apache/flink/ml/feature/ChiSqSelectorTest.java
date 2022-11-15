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

import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.feature.chisqselector.ChiSqSelector;
import org.apache.flink.ml.feature.chisqselector.ChiSqSelectorModel;
import org.apache.flink.ml.feature.chisqselector.ChiSqSelectorModelData;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.apache.flink.ml.feature.chisqselector.ChiSqSelectorParams.FDR_TYPE;
import static org.apache.flink.ml.feature.chisqselector.ChiSqSelectorParams.FPR_TYPE;
import static org.apache.flink.ml.feature.chisqselector.ChiSqSelectorParams.FWE_TYPE;
import static org.apache.flink.ml.feature.chisqselector.ChiSqSelectorParams.NUM_TOP_FEATURES_TYPE;
import static org.apache.flink.ml.feature.chisqselector.ChiSqSelectorParams.PERCENTILE_TYPE;
import static org.apache.flink.table.api.Expressions.$;
import static org.apache.flink.table.api.Expressions.nullOf;
import static org.junit.Assert.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/** Tests {@link ChiSqSelector} and {@link ChiSqSelectorModel}. */
public class ChiSqSelectorTest extends AbstractTestBase {
    private static final double EPS = 1.0e-5;

    private StreamTableEnvironment tEnv;
    private Table inputTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            0.0,
                            Vectors.sparse(
                                    6, new int[] {0, 1, 3, 4}, new double[] {6.0, 7.0, 7.0, 6.0})),
                    Row.of(
                            1.0,
                            Vectors.sparse(
                                    6, new int[] {1, 2, 4, 5}, new double[] {9.0, 6.0, 5.0, 9.0})),
                    Row.of(
                            1.0,
                            Vectors.sparse(
                                    6, new int[] {1, 2, 4, 5}, new double[] {9.0, 3.0, 5.0, 5.0})),
                    Row.of(1.0, Vectors.dense(0.0, 9.0, 8.0, 5.0, 6.0, 4.0)),
                    Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 5.0, 4.0, 4.0)),
                    Row.of(2.0, Vectors.dense(8.0, 9.0, 6.0, 4.0, 0.0, 0.0)));

    @Before
    public void before() {
        Configuration config = new Configuration();
        config.set(ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);
        env.setParallelism(4);
        env.enableCheckpointing(100);
        env.setRestartStrategy(RestartStrategies.noRestart());
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> inputStream =
                env.fromCollection(
                        INPUT_DATA, new RowTypeInfo(Types.DOUBLE, VectorTypeInfo.INSTANCE));
        inputTable = tEnv.fromDataStream(inputStream).as("label", "features");
    }

    @Test
    public void testParam() {
        ChiSqSelector selector = new ChiSqSelector();

        assertEquals(NUM_TOP_FEATURES_TYPE, selector.getSelectorType());
        assertEquals(50, selector.getNumTopFeatures());
        assertEquals(0.1, selector.getPercentile(), EPS);
        assertEquals(0.05, selector.getFpr(), EPS);
        assertEquals(0.05, selector.getFdr(), EPS);
        assertEquals(0.05, selector.getFwe(), EPS);
        assertEquals("features", selector.getFeaturesCol());
        assertEquals("label", selector.getLabelCol());
        assertEquals("output", selector.getOutputCol());

        selector.setSelectorType(PERCENTILE_TYPE)
                .setNumTopFeatures(10)
                .setPercentile(0.5)
                .setFpr(0.1)
                .setFdr(0.1)
                .setFwe(0.1)
                .setFeaturesCol("test_features")
                .setLabelCol("test_label")
                .setOutputCol("test_output");

        assertEquals(PERCENTILE_TYPE, selector.getSelectorType());
        assertEquals(10, selector.getNumTopFeatures());
        assertEquals(0.5, selector.getPercentile(), EPS);
        assertEquals(0.1, selector.getFpr(), EPS);
        assertEquals(0.1, selector.getFdr(), EPS);
        assertEquals(0.1, selector.getFwe(), EPS);
        assertEquals("test_features", selector.getFeaturesCol());
        assertEquals("test_label", selector.getLabelCol());
        assertEquals("test_output", selector.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        inputTable =
                inputTable.select(
                        $("label").as("test_label"),
                        $("features").as("test_features"),
                        nullOf(DataTypes.INT()).as("dummy_input"));
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("test_features")
                        .setLabelCol("test_label")
                        .setOutputCol("test_output");
        Table outputTable = selector.fit(inputTable).transform(inputTable)[0];

        assertEquals(
                Arrays.asList("test_label", "test_features", "dummy_input", "test_output"),
                outputTable.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredictWithNumTopFeatures() {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(NUM_TOP_FEATURES_TYPE)
                        .setNumTopFeatures(1);

        Table outputTable = selector.fit(inputTable).transform(inputTable)[0];

        verifyOutput(outputTable, selector.getFeaturesCol(), selector.getOutputCol(), 0);
    }

    @Test
    public void testFitAndPredictWithPercentile() {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(PERCENTILE_TYPE)
                        .setPercentile(0.17);

        Table outputTable = selector.fit(inputTable).transform(inputTable)[0];

        verifyOutput(outputTable, selector.getFeaturesCol(), selector.getOutputCol(), 0);
    }

    @Test
    public void testFitAndPredictWithFPR() {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(FPR_TYPE)
                        .setFpr(0.02);

        Table outputTable = selector.fit(inputTable).transform(inputTable)[0];

        verifyOutput(outputTable, selector.getFeaturesCol(), selector.getOutputCol(), 0);
    }

    @Test
    public void testFitAndPredictWithFDR() {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(FDR_TYPE)
                        .setFdr(0.12);

        Table outputTable = selector.fit(inputTable).transform(inputTable)[0];

        verifyOutput(outputTable, selector.getFeaturesCol(), selector.getOutputCol(), 0);
    }

    @Test
    public void testFitAndPredictWithFWE() {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(FWE_TYPE)
                        .setFwe(0.12);

        Table outputTable = selector.fit(inputTable).transform(inputTable)[0];

        verifyOutput(outputTable, selector.getFeaturesCol(), selector.getOutputCol(), 0);
    }

    @Test
    public void testSaveLoadAndPredict() throws Exception {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(NUM_TOP_FEATURES_TYPE)
                        .setNumTopFeatures(1);

        selector =
                TestUtils.saveAndReload(
                        tEnv, selector, TEMPORARY_FOLDER.newFolder().getAbsolutePath());

        ChiSqSelectorModel selectorModel = selector.fit(inputTable);

        selectorModel =
                TestUtils.saveAndReload(
                        tEnv, selectorModel, TEMPORARY_FOLDER.newFolder().getAbsolutePath());

        Table outputTable = selectorModel.transform(inputTable)[0];

        verifyOutput(outputTable, selector.getFeaturesCol(), selector.getOutputCol(), 0);
    }

    @Test
    public void testGetModelData() throws Exception {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(NUM_TOP_FEATURES_TYPE)
                        .setNumTopFeatures(1);

        ChiSqSelectorModel selectorModel = selector.fit(inputTable);

        Table modelDataTable = selectorModel.getModelData()[0];

        assertEquals(
                Collections.singletonList("selectedFeatures"),
                modelDataTable.getResolvedSchema().getColumnNames());

        List<ChiSqSelectorModelData> modelDataList =
                IteratorUtils.toList(
                        ChiSqSelectorModelData.getModelDataStream(modelDataTable)
                                .executeAndCollect());
        assertEquals(1, modelDataList.size());

        ChiSqSelectorModelData modelData = modelDataList.get(0);
        assertArrayEquals(new int[] {0}, modelData.selectedFeatureIndices);
    }

    @Test
    public void testSetModelData() {
        ChiSqSelector selector =
                new ChiSqSelector()
                        .setFeaturesCol("features")
                        .setLabelCol("label")
                        .setOutputCol("prediction")
                        .setSelectorType(NUM_TOP_FEATURES_TYPE)
                        .setNumTopFeatures(1);

        ChiSqSelectorModel selectorModel = selector.fit(inputTable);

        ChiSqSelectorModel newSelectorModel = new ChiSqSelectorModel();
        ReadWriteUtils.updateExistingParams(newSelectorModel, selectorModel.getParamMap());
        newSelectorModel.setModelData(selectorModel.getModelData());

        Table outputTable = newSelectorModel.transform(inputTable)[0];

        verifyOutput(outputTable, selector.getFeaturesCol(), selector.getOutputCol(), 0);
    }

    private void verifyOutput(
            Table outputTable, String featuresCol, String outputCol, int... selectedIndices) {
        List<Row> outputValues = IteratorUtils.toList(outputTable.execute().collect());
        for (Row row : outputValues) {
            Vector input = row.getFieldAs(featuresCol);
            double[] expected = new double[selectedIndices.length];
            for (int i = 0; i < selectedIndices.length; i++) {
                expected[i] = input.get(selectedIndices[i]);
            }
            Vector actual = row.getFieldAs(outputCol);
            assertArrayEquals(expected, actual.toArray());
        }
    }
}
