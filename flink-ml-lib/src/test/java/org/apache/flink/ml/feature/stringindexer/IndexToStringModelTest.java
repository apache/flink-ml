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
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests the {@link IndexToStringModel}. */
public class IndexToStringModelTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private Table predictTable;
    private Table modelTable;
    private Table predictTableWithUnseenValues;

    private final List<Row> expectedPrediction =
            Arrays.asList(Row.of(0, 3, "a", "2.0"), Row.of(1, 2, "b", "1.0"));
    private final String[][] stringArrays =
            new String[][] {{"a", "b", "c", "d"}, {"-1.0", "0.0", "1.0", "2.0"}};

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        modelTable =
                tEnv.fromDataStream(env.fromElements(new StringIndexerModelData(stringArrays)))
                        .as("stringArrays");
        predictTable =
                tEnv.fromDataStream(env.fromCollection(Arrays.asList(Row.of(0, 3), Row.of(1, 2))))
                        .as("inputCol1", "inputCol2");
        predictTableWithUnseenValues =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        Arrays.asList(Row.of(0, 3), Row.of(1, 2), Row.of(4, 1))))
                        .as("inputCol1", "inputCol2");
    }

    @Test
    public void testOutputSchema() {
        IndexToStringModel indexToStringModel =
                new IndexToStringModel()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setModelData(modelTable);
        Table output = indexToStringModel.transform(predictTable)[0];

        assertEquals(
                Arrays.asList("inputCol1", "inputCol2", "outputCol1", "outputCol2"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testInputWithUnseenValues() {
        IndexToStringModel indexToStringModel =
                new IndexToStringModel()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setModelData(modelTable);
        Table output = indexToStringModel.transform(predictTableWithUnseenValues)[0];

        try {
            IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "The input contains unseen index: 4.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testPredict() throws Exception {
        IndexToStringModel indexToStringModel =
                new IndexToStringModel()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setModelData(modelTable);
        Table output = indexToStringModel.transform(predictTable)[0];

        List<Row> predictedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        StringIndexerTest.verifyPredictionResult(expectedPrediction, predictedResult);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testSaveLoadAndPredict() throws Exception {
        IndexToStringModel model =
                new IndexToStringModel()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setModelData(modelTable);
        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        tempFolder.newFolder().getAbsolutePath(),
                        IndexToStringModel::load);

        assertEquals(
                Collections.singletonList("stringArrays"),
                model.getModelData()[0].getResolvedSchema().getColumnNames());

        Table output = model.transform(predictTable)[0];

        List<Row> predictedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        StringIndexerTest.verifyPredictionResult(expectedPrediction, predictedResult);
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        IndexToStringModel model =
                new IndexToStringModel()
                        .setInputCols("inputCol1", "inputCol2")
                        .setOutputCols("outputCol1", "outputCol2")
                        .setModelData(modelTable);

        List<StringIndexerModelData> collectedModelData =
                (List<StringIndexerModelData>)
                        (IteratorUtils.toList(
                                StringIndexerModelData.getModelDataStream(model.getModelData()[0])
                                        .executeAndCollect()));

        assertEquals(1, collectedModelData.size());

        StringIndexerModelData modelData = collectedModelData.get(0);
        assertEquals(2, modelData.stringArrays.length);
        assertArrayEquals(stringArrays[0], modelData.stringArrays[0]);
        assertArrayEquals(stringArrays[1], modelData.stringArrays[1]);
    }
}
