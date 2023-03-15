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

import org.apache.flink.ml.feature.binarizer.Binarizer;
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

/** Tests {@link Binarizer}. */
public class BinarizerTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(
                            1,
                            Vectors.dense(1, 2),
                            Vectors.sparse(17, new int[] {0, 3, 9}, new double[] {1.0, 2.0, 7.0})),
                    Row.of(
                            2,
                            Vectors.dense(2, 1),
                            Vectors.sparse(17, new int[] {0, 2, 14}, new double[] {5.0, 4.0, 1.0})),
                    Row.of(
                            3,
                            Vectors.dense(5, 18),
                            Vectors.sparse(
                                    17, new int[] {0, 11, 12}, new double[] {2.0, 4.0, 4.0})));

    private static final Double[] EXPECTED_VALUE_OUTPUT = new Double[] {0.0, 1.0, 1.0};

    private static final List<Vector> EXPECTED_DENSE_OUTPUT =
            Arrays.asList(
                    Vectors.dense(0.0, 1.0), Vectors.dense(1.0, 0.0), Vectors.dense(1.0, 1.0));

    private static final List<Vector> EXPECTED_SPARSE_OUTPUT =
            Arrays.asList(
                    Vectors.sparse(17, new int[] {9}, new double[] {1.0}),
                    Vectors.sparse(17, new int[] {0, 2}, new double[] {1.0, 1.0}),
                    Vectors.sparse(17, new int[] {11, 12}, new double[] {1.0, 1.0}));

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT_DATA);
        inputDataTable = tEnv.fromDataStream(dataStream).as("f0", "f1", "f2");
    }

    private void verifyOutputResult(Table output, String[] outputCols) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) output).getTableEnvironment();
        DataStream<Row> stream = tEnv.toDataStream(output);

        List<Row> results = IteratorUtils.toList(stream.executeAndCollect());
        List<Double> doubleValues = new ArrayList<>(results.size());
        List<Vector> sparseVectorValues = new ArrayList<>(results.size());
        List<Vector> denseVectorValues = new ArrayList<>(results.size());
        for (Row row : results) {
            doubleValues.add(row.getFieldAs(outputCols[0]));
            denseVectorValues.add(row.getFieldAs(outputCols[1]));
            sparseVectorValues.add(row.getFieldAs(outputCols[2]));
        }
        doubleValues.sort(Double::compare);
        assertArrayEquals(EXPECTED_VALUE_OUTPUT, doubleValues.toArray());
        compareResultCollections(EXPECTED_DENSE_OUTPUT, denseVectorValues, TestUtils::compare);
        compareResultCollections(EXPECTED_SPARSE_OUTPUT, sparseVectorValues, TestUtils::compare);
    }

    @Test
    public void testParam() {
        Binarizer binarizer =
                new Binarizer()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCols("of0", "of1", "of2")
                        .setThresholds(0.0, 0.0, 0.0);

        assertArrayEquals(new String[] {"f0", "f1", "f2"}, binarizer.getInputCols());
        assertArrayEquals(new String[] {"of0", "of1", "of2"}, binarizer.getOutputCols());
        assertArrayEquals(new Double[] {0.0, 0.0, 0.0}, binarizer.getThresholds());
    }

    @Test
    public void testOutputSchema() {
        Binarizer binarizer =
                new Binarizer()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCols("of0", "of1", "of2")
                        .setThresholds(0.0, 0.0, 0.0);

        Table output = binarizer.transform(inputDataTable)[0];

        assertEquals(
                Arrays.asList("f0", "f1", "f2", "of0", "of1", "of2"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        Binarizer binarizer =
                new Binarizer()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCols("of0", "of1", "of2")
                        .setThresholds(1.0, 1.5, 2.5);

        Binarizer loadedBinarizer =
                TestUtils.saveAndReload(
                        tEnv,
                        binarizer,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        Binarizer::load);

        Table output = loadedBinarizer.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedBinarizer.getOutputCols());
    }
}
