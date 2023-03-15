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

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.feature.bucketizer.Bucketizer;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests the {@link Bucketizer}. */
public class BucketizerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamTableEnvironment tEnv;
    private Table inputTable;

    private static final List<Row> inputData =
            Arrays.asList(
                    Row.of(1, -0.5, 0.0, 1.0, 0.0),
                    Row.of(2, Double.NEGATIVE_INFINITY, 1.0, Double.POSITIVE_INFINITY, 1.0),
                    Row.of(3, Double.NaN, -0.5, -0.5, 2.0));

    private static final Double[][] splitsArray =
            new Double[][] {
                new Double[] {-0.5, 0.0, 0.5},
                new Double[] {-1.0, 0.0, 2.0},
                new Double[] {Double.NEGATIVE_INFINITY, 10.0, Double.POSITIVE_INFINITY},
                new Double[] {Double.NEGATIVE_INFINITY, 1.5, Double.POSITIVE_INFINITY}
            };

    private final List<Row> expectedKeepResult =
            Arrays.asList(
                    Row.of(1, 0.0, 1.0, 0.0, 0.0),
                    Row.of(2, 2.0, 1.0, 1.0, 0.0),
                    Row.of(3, 2.0, 0.0, 0.0, 1.0));

    private final List<Row> expectedSkipResult =
            Collections.singletonList(Row.of(1, 0.0, 1.0, 0.0, 0.0));

    @Before
    public void before() {
        StreamExecutionEnvironment env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        inputTable =
                tEnv.fromDataStream(env.fromCollection(inputData)).as("id", "f1", "f2", "f3", "f4");
    }

    @SuppressWarnings("all")
    private void verifyOutputResult(Table output, String[] outputCols, List<Row> expectedResult)
            throws Exception {
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<Row> result = new ArrayList<>(collectedResult.size());

        for (Row r : collectedResult) {
            Row outRow = new Row(outputCols.length + 1);
            outRow.setField(0, r.getField("id"));
            for (int i = 0; i < outputCols.length; i++) {
                outRow.setField(i + 1, r.getField(outputCols[i]));
            }
            result.add(outRow);
        }

        compareResultCollections(
                expectedResult, result, Comparator.comparingInt(r -> ((Integer) r.getField(0))));
    }

    @Test
    public void testParam() {
        Bucketizer bucketizer = new Bucketizer();
        assertEquals(HasHandleInvalid.ERROR_INVALID, bucketizer.getHandleInvalid());

        bucketizer
                .setInputCols("f1", "f2", "f3", "f4")
                .setOutputCols("o1", "o2", "o3", "o4")
                .setHandleInvalid(HasHandleInvalid.SKIP_INVALID)
                .setSplitsArray(splitsArray);
        assertArrayEquals(new String[] {"f1", "f2", "f3", "f4"}, bucketizer.getInputCols());
        assertArrayEquals(new String[] {"o1", "o2", "o3", "o4"}, bucketizer.getOutputCols());
        assertEquals(HasHandleInvalid.SKIP_INVALID, bucketizer.getHandleInvalid());

        Double[][] setSplitsArray = bucketizer.getSplitsArray();
        assertEquals(splitsArray.length, setSplitsArray.length);
        for (int i = 0; i < splitsArray.length; i++) {
            assertArrayEquals(splitsArray[i], setSplitsArray[i]);
        }
    }

    @Test
    public void testOutputSchema() {
        Bucketizer bucketizer =
                new Bucketizer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4")
                        .setHandleInvalid(HasHandleInvalid.SKIP_INVALID)
                        .setSplitsArray(splitsArray);
        Table output = bucketizer.transform(inputTable)[0];
        assertEquals(
                Arrays.asList("id", "f1", "f2", "f3", "f4", "o1", "o2", "o3", "o4"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransform() throws Exception {
        Bucketizer bucketizer =
                new Bucketizer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4")
                        .setSplitsArray(splitsArray);

        Table output;

        // Tests skip.
        bucketizer.setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        output = bucketizer.transform(inputTable)[0];
        verifyOutputResult(output, bucketizer.getOutputCols(), expectedSkipResult);

        // Tests keep.
        bucketizer.setHandleInvalid(HasHandleInvalid.KEEP_INVALID);
        output = bucketizer.transform(inputTable)[0];
        verifyOutputResult(output, bucketizer.getOutputCols(), expectedKeepResult);

        // Tests error.
        bucketizer.setHandleInvalid(HasHandleInvalid.ERROR_INVALID);
        output = bucketizer.transform(inputTable)[0];
        try {
            IteratorUtils.toList(output.execute().collect());
            fail();
        } catch (Throwable e) {
            assertEquals(
                    "The input contains invalid value. See "
                            + HasHandleInvalid.HANDLE_INVALID
                            + " parameter for more options.",
                    ExceptionUtils.getRootCause(e).getMessage());
        }
    }

    @Test
    public void testInputTypeConversion() throws Exception {
        inputTable = TestUtils.convertDataTypesToSparseInt(tEnv, inputTable);
        assertArrayEquals(
                new Class<?>[] {
                    Integer.class, Integer.class, Integer.class, Integer.class, Integer.class
                },
                TestUtils.getColumnDataTypes(inputTable));

        Bucketizer bucketizer =
                new Bucketizer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4")
                        .setSplitsArray(splitsArray);

        bucketizer.setHandleInvalid(HasHandleInvalid.SKIP_INVALID);
        Table output = bucketizer.transform(inputTable)[0];

        List<Row> expectedResult =
                Arrays.asList(Row.of(1, 1.0, 1.0, 0.0, 0.0), Row.of(3, 1.0, 1.0, 0.0, 1.0));
        verifyOutputResult(output, bucketizer.getOutputCols(), expectedResult);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        Bucketizer bucketizer =
                new Bucketizer()
                        .setInputCols("f1", "f2", "f3", "f4")
                        .setOutputCols("o1", "o2", "o3", "o4")
                        .setHandleInvalid(HasHandleInvalid.KEEP_INVALID)
                        .setSplitsArray(splitsArray);
        Bucketizer loadedBucketizer =
                TestUtils.saveAndReload(
                        tEnv,
                        bucketizer,
                        tempFolder.newFolder().getAbsolutePath(),
                        Bucketizer::load);
        Table output = loadedBucketizer.transform(inputTable)[0];
        verifyOutputResult(output, loadedBucketizer.getOutputCols(), expectedKeepResult);
    }
}
