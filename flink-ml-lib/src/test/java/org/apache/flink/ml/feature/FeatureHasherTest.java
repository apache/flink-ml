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

import org.apache.flink.ml.feature.featurehasher.FeatureHasher;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link FeatureHasher}. */
public class FeatureHasherTest extends AbstractTestBase {

    private StreamTableEnvironment tEnv;
    private Table inputDataTable;

    private static final List<Row> INPUT_DATA =
            Arrays.asList(Row.of(0, "a", 1.0, true), Row.of(1, "c", 1.0, false));

    private static final SparseIntDoubleVector EXPECTED_OUTPUT_DATA_1 =
            Vectors.sparse(1000, new int[] {607, 635, 913}, new double[] {1.0, 1.0, 1.0});
    private static final SparseIntDoubleVector EXPECTED_OUTPUT_DATA_2 =
            Vectors.sparse(1000, new int[] {242, 869, 913}, new double[] {1.0, 1.0, 1.0});

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> dataStream = env.fromCollection(INPUT_DATA);
        inputDataTable = tEnv.fromDataStream(dataStream).as("id", "f0", "f1", "f2");
    }

    private void verifyOutputResult(Table output, String outputCol) throws Exception {
        DataStream<Row> dataStream = tEnv.toDataStream(output);
        List<Row> results = IteratorUtils.toList(dataStream.executeAndCollect());
        assertEquals(2, results.size());
        for (Row result : results) {
            if (result.getField(0) == (Object) 0) {
                assertEquals(EXPECTED_OUTPUT_DATA_1, result.getField(outputCol));
            } else if (result.getField(0) == (Object) 1) {
                assertEquals(EXPECTED_OUTPUT_DATA_2, result.getField(outputCol));
            } else {
                throw new RuntimeException("unknown output value.");
            }
        }
    }

    @Test
    public void testParam() {
        FeatureHasher featureHasher = new FeatureHasher();
        assertEquals("output", featureHasher.getOutputCol());
        assertArrayEquals(new String[] {}, featureHasher.getCategoricalCols());
        assertEquals(262144, featureHasher.getNumFeatures());
        featureHasher
                .setInputCols("f0", "f1", "f2")
                .setOutputCol("vec")
                .setCategoricalCols("f0", "f2")
                .setNumFeatures(1000);
        assertArrayEquals(new String[] {"f0", "f1", "f2"}, featureHasher.getInputCols());
        assertEquals("vec", featureHasher.getOutputCol());
        assertArrayEquals(new String[] {"f0", "f2"}, featureHasher.getCategoricalCols());
        assertEquals(1000, featureHasher.getNumFeatures());
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        FeatureHasher featureHash =
                new FeatureHasher()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCol("vec")
                        .setCategoricalCols("f0", "f2")
                        .setNumFeatures(1000);
        FeatureHasher loadedFeatureHasher =
                TestUtils.saveAndReload(
                        tEnv,
                        featureHash,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        FeatureHasher::load);
        Table output = loadedFeatureHasher.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedFeatureHasher.getOutputCol());
    }

    @Test
    public void testCategoricalColsNotSet() throws Exception {
        FeatureHasher featureHash =
                new FeatureHasher()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCol("vec")
                        .setNumFeatures(1000);
        FeatureHasher loadedFeatureHasher =
                TestUtils.saveAndReload(
                        tEnv,
                        featureHash,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        FeatureHasher::load);
        Table output = loadedFeatureHasher.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedFeatureHasher.getOutputCol());
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        inputDataTable = TestUtils.convertDataTypesToSparseInt(tEnv, inputDataTable);
        assertArrayEquals(
                new Class<?>[] {Integer.class, String.class, Integer.class, Boolean.class},
                TestUtils.getColumnDataTypes(inputDataTable));

        FeatureHasher featureHash =
                new FeatureHasher()
                        .setInputCols("f0", "f1", "f2")
                        .setOutputCol("vec")
                        .setCategoricalCols("f0", "f2")
                        .setNumFeatures(1000);
        FeatureHasher loadedFeatureHasher =
                TestUtils.saveAndReload(
                        tEnv,
                        featureHash,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        FeatureHasher::load);
        Table output = loadedFeatureHasher.transform(inputDataTable)[0];
        verifyOutputResult(output, loadedFeatureHasher.getOutputCol());
    }
}
