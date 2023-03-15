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

import org.apache.flink.ml.feature.interaction.Interaction;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.SparseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
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
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link Interaction}. */
public class InteractionTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            1,
                            Vectors.dense(1, 2),
                            Vectors.dense(3, 4),
                            Vectors.sparse(17, new int[] {0, 3, 9}, new double[] {1.0, 2.0, 7.0})),
                    Row.of(
                            2,
                            Vectors.dense(2, 8),
                            Vectors.dense(3, 4, 5),
                            Vectors.sparse(17, new int[] {0, 2, 14}, new double[] {5.0, 4.0, 1.0})),
                    Row.of(3, null, null, null));

    private static final List<Vector> EXPECTED_DENSE_OUTPUT =
            Arrays.asList(
                    new DenseVector(new double[] {3.0, 4.0, 6.0, 8.0}),
                    new DenseVector(new double[] {12.0, 16.0, 20.0, 48.0, 64.0, 80.0}));

    private static final List<Vector> EXPECTED_SPARSE_OUTPUT =
            Arrays.asList(
                    new SparseVector(
                            68,
                            new int[] {0, 3, 9, 17, 20, 26, 34, 37, 43, 51, 54, 60},
                            new double[] {
                                3.0, 6.0, 21.0, 4.0, 8.0, 28.0, 6.0, 12.0, 42.0, 8.0, 16.0, 56.0
                            }),
                    new SparseVector(
                            102,
                            new int[] {
                                0, 2, 14, 17, 19, 31, 34, 36, 48, 51, 53, 65, 68, 70, 82, 85, 87, 99
                            },
                            new double[] {
                                60.0, 48.0, 12.0, 80.0, 64.0, 16.0, 100.0, 80.0, 20.0, 240.0, 192.0,
                                48.0, 320.0, 256.0, 64.0, 400.0, 320.0, 80.0
                            }));

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT_DATA);
        inputDataTable = tEnv.fromDataStream(dataStream).as("f0", "f1", "f2", "f3");
    }

    private void verifyOutputResult(Table output, String outputCol, List<Vector> expectedData)
            throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Row> stream = tEnv.toDataStream(output);

        List<Row> results = IteratorUtils.toList(stream.executeAndCollect());
        List<Vector> resultVec = new ArrayList<>(results.size());
        for (Row row : results) {
            if (row.getField(outputCol) != null) {
                resultVec.add(row.getFieldAs(outputCol));
            }
        }
        compareResultCollections(expectedData, resultVec, TestUtils::compare);
    }

    @Test
    public void testParam() {
        Interaction interaction = new Interaction();
        assertEquals("output", interaction.getOutputCol());

        interaction.setInputCols("f0", "f1", "f2").setOutputCol("interactionVecVec");

        assertArrayEquals(new String[] {"f0", "f1", "f2"}, interaction.getInputCols());
        assertEquals("interactionVecVec", interaction.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        Interaction interaction =
                new Interaction().setInputCols("f0", "f1", "f2", "f3").setOutputCol("outputVec");

        Table output = interaction.transform(inputDataTable)[0];

        assertEquals(
                Arrays.asList("f0", "f1", "f2", "f3", "outputVec"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testSaveLoadAndTransformSparse() throws Exception {
        Interaction interaction =
                new Interaction()
                        .setInputCols("f0", "f1", "f2", "f3")
                        .setOutputCol("interactionVecVec");

        Interaction loadedInteraction =
                TestUtils.saveAndReload(
                        tEnv,
                        interaction,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        Interaction::load);

        Table output = loadedInteraction.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedInteraction.getOutputCol(), EXPECTED_SPARSE_OUTPUT);
    }

    @Test
    public void testSaveLoadAndTransformDense() throws Exception {
        Interaction interaction =
                new Interaction().setInputCols("f0", "f1", "f2").setOutputCol("interactionVecVec");

        Interaction loadedInteraction =
                TestUtils.saveAndReload(
                        tEnv,
                        interaction,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        Interaction::load);

        Table output = loadedInteraction.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedInteraction.getOutputCol(), EXPECTED_DENSE_OUTPUT);
    }
}
