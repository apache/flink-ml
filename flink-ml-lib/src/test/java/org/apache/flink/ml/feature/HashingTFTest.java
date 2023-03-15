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

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.feature.hashingtf.HashingTF;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Expressions;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Tests {@link HashingTF}. */
public class HashingTFTest extends AbstractTestBase {
    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table inputDataTable;

    private static final List<Row> INPUT =
            Arrays.asList(
                    Row.of(Arrays.asList("HashingTFTest", "Hashing", "Term", "Frequency", "Test")),
                    Row.of(Arrays.asList("HashingTFTest", "Hashing", "Hashing", "Test", "Test")));

    private static final List<Row> EXPECTED_OUTPUT =
            Arrays.asList(
                    Row.of(
                            Vectors.sparse(
                                    262144,
                                    new int[] {67564, 89917, 113827, 131486, 228971},
                                    new double[] {1.0, 1.0, 1.0, 1.0, 1.0})),
                    Row.of(
                            Vectors.sparse(
                                    262144,
                                    new int[] {67564, 131486, 228971},
                                    new double[] {1.0, 2.0, 2.0})));

    private static final List<Row> EXPECTED_BINARY_OUTPUT =
            Arrays.asList(
                    Row.of(
                            Vectors.sparse(
                                    262144,
                                    new int[] {67564, 89917, 113827, 131486, 228971},
                                    new double[] {1.0, 1.0, 1.0, 1.0, 1.0})),
                    Row.of(
                            Vectors.sparse(
                                    262144,
                                    new int[] {67564, 131486, 228971},
                                    new double[] {1.0, 1.0, 1.0})));

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT, Types.ROW(Types.LIST(Types.STRING)));
        inputDataTable = tEnv.fromDataStream(dataStream).as("input");
    }

    @Test
    public void testParam() {
        HashingTF hashingTF = new HashingTF();
        assertEquals("input", hashingTF.getInputCol());
        assertFalse(hashingTF.getBinary());
        assertEquals(262144, hashingTF.getNumFeatures());
        assertEquals("output", hashingTF.getOutputCol());

        hashingTF
                .setInputCol("testInputCol")
                .setBinary(true)
                .setNumFeatures(1024)
                .setOutputCol("testOutputCol");

        assertEquals("testInputCol", hashingTF.getInputCol());
        assertTrue(hashingTF.getBinary());
        assertEquals(1024, hashingTF.getNumFeatures());
        assertEquals("testOutputCol", hashingTF.getOutputCol());
    }

    @Test
    public void testOutputSchema() {
        HashingTF hashingTF = new HashingTF();
        inputDataTable =
                tEnv.fromDataStream(env.fromElements(Row.of(Arrays.asList(""), Arrays.asList(""))))
                        .as("input", "dummyInput");

        Table output = hashingTF.transform(inputDataTable)[0];
        assertEquals(
                Arrays.asList(hashingTF.getInputCol(), "dummyInput", hashingTF.getOutputCol()),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        HashingTF hashingTF = new HashingTF();
        Table output;

        // Tests non-binary.
        output = hashingTF.transform(inputDataTable)[0];
        verifyOutputResult(output, hashingTF.getOutputCol(), EXPECTED_OUTPUT);

        // Tests binary.
        hashingTF.setBinary(true);
        output = hashingTF.transform(inputDataTable)[0];
        verifyOutputResult(output, hashingTF.getOutputCol(), EXPECTED_BINARY_OUTPUT);
    }

    @Test
    public void testTransformArrayData() throws Exception {
        HashingTF hashingTF = new HashingTF();
        inputDataTable =
                tEnv.fromDataStream(
                                env.fromElements(
                                        new String[] {
                                            "HashingTFTest", "Hashing", "Term", "Frequency", "Test"
                                        },
                                        new String[] {
                                            "HashingTFTest", "Hashing", "Hashing", "Test", "Test"
                                        }))
                        .as("input");

        Table output = hashingTF.transform(inputDataTable)[0];
        verifyOutputResult(output, hashingTF.getOutputCol(), EXPECTED_OUTPUT);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        HashingTF hashingTF = new HashingTF();
        HashingTF loadedHashingTF =
                TestUtils.saveAndReload(
                        tEnv,
                        hashingTF,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        HashingTF::load);

        Table output = loadedHashingTF.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedHashingTF.getOutputCol(), EXPECTED_OUTPUT);
    }

    private void verifyOutputResult(Table output, String outputCol, List<Row> expectedOutput)
            throws Exception {
        DataStream<Row> dataStream = tEnv.toDataStream(output.select(Expressions.$(outputCol)));
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(expectedOutput.size(), results.size());

        results.sort(Comparator.comparingInt(o -> o.getField(0).hashCode()));
        expectedOutput.sort(Comparator.comparingInt(o -> o.getField(0).hashCode()));
        for (int i = 0; i < expectedOutput.size(); i++) {
            assertEquals(expectedOutput.get(i).getField(0), results.get(i).getField(0));
        }
    }
}
