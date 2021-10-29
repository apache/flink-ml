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

package org.apache.flink.ml.dataproc.stringindexer;

import org.apache.commons.collections.IteratorUtils;
import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.RestOptions;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.ml.util.TableUtils;
import org.apache.flink.streaming.api.environment.ExecutionCheckpointingOptions;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.types.AbstractDataType;
import org.apache.flink.types.Row;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.*;

public class MultiStringIndexerTest {
    @Rule
    public ExpectedException thrown = ExpectedException.none();
    @Rule public TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    DataTypes.Field[] fields;
    Row[] inputData;
    Row[] inputDataUnseenToken;
    Set<Row[]> expectedOutput;
    String[] expectedOutputNames;
    AbstractDataType[] expectedOutputTypes;
    String[] selectedCols;
    String[] outputCols;
    String[] reservedCols;
    String handleInvalid;
    String errorMessage;

    @Before
    public void Setup() throws IOException {
        env = StreamExecutionEnvironment.createLocalEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        env.setRuntimeMode(RuntimeExecutionMode.BATCH);
        Configuration configuration = new Configuration();
        configuration.set(RestOptions.PORT, 18082);
        configuration.set(
                IterationOptions.DATA_CACHE_PATH,
                "file://" + tempFolder.newFolder().getAbsolutePath());
        configuration.set(
                ExecutionCheckpointingOptions.ENABLE_CHECKPOINTS_AFTER_TASKS_FINISH, true);
        env.getConfig().setGlobalJobParameters(configuration);

        fields = new DataTypes.Field[]{
                DataTypes.FIELD("f0", DataTypes.STRING()),
                DataTypes.FIELD("f1", DataTypes.INT()),
        };

        inputData = new Row[] {
                Row.of("a", 1L),
                Row.of(null, 1L),
                Row.of("b", 1L),
                Row.of("b", 3L),
        };

        expectedOutput = new HashSet<>();

        expectedOutput.add(new Row[] {
                Row.of("a", 0L, 0L),
                Row.of(null, null, 0L),
                Row.of("b", 1L, 0L),
                Row.of("b", 1L, 1L),
        });

        expectedOutput.add(new Row[] {
                Row.of("a", 1L, 0L),
                Row.of(null, null, 0L),
                Row.of("b", 0L, 0L),
                Row.of("b", 0L, 1L),
        });

        expectedOutput.add(new Row[] {
                Row.of("a", 0L, 1L),
                Row.of(null, null, 1L),
                Row.of("b", 1L, 1L),
                Row.of("b", 1L, 0L),
        });

        expectedOutput.add(new Row[] {
                Row.of("a", 1L, 1L),
                Row.of(null, null, 1L),
                Row.of("b", 0L, 1L),
                Row.of("b", 0L, 0L),
        });

        expectedOutputNames = new String[]{
                "f0",
                "f0_index",
                "f1_index",
        };

        expectedOutputTypes = new AbstractDataType[]{
                DataTypes.STRING(),
                DataTypes.BIGINT(),
                DataTypes.BIGINT()
        };

        inputDataUnseenToken = new Row[] {
                Row.of("c", 4L),
                Row.of(null, 1L),
                Row.of("b", 1L),
                Row.of("b", 3L),
        };

        selectedCols = new String[]{"f0", "f1"};
        outputCols = new String[]{"f0_index", "f1_index"};
        reservedCols = new String[]{"f0"};
        handleInvalid = "SKIP";
    }


    @Test
    public void testMultiStringIndexer() {
        errorMessage = "normal test for MultiStringIndexer";
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test
    public void testSelectedColsNotString() {
        errorMessage = "MultiStringIndexer should also deal with selected columns whose type is not string.";
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test(expected = Exception.class)
    public void testSelectedNotMatchOutputCols() {
        errorMessage = "MultiStringIndexer should throw Exception when length of selected columns and output columns does not match.";
        outputCols = new String[]{"f1_index"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test(expected = Exception.class)
    public void testNullSelectedCols() {
        errorMessage = "MultiStringIndexer should throw Exception when selected columns is null.";
        selectedCols = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test (expected = Exception.class)
    public void testMissingSelectedCols() {
        errorMessage = "MultiStringIndexer should throw Exception when predict data does not contain selected columns.";
        selectedCols = new String[]{"f-1", "f1"};
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);

    }

    @Test (expected = Exception.class)
    public void testNullOutputCols() {
        errorMessage = "MultiStringIndexer should throw Exception when output columns is null.";
        outputCols = null;
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test
    public void testNullReservedCols() {
        errorMessage = "MultiStringIndexer should run without throwing Exception when reserved columns is null.";
        reservedCols = null;
        expectedOutput = new HashSet<>();
        expectedOutput.add(new Row[] {
                Row.of(1L, 0L),
                Row.of(null, 0L),
                Row.of(0L, 0L),
                Row.of(0L, 1L),
        });
        expectedOutput.add(new Row[] {
                Row.of(1L, 1L),
                Row.of(null, 1L),
                Row.of(0L, 1L),
                Row.of(0L, 0L),
        });
        expectedOutput.add(new Row[] {
                Row.of(0L, 0L),
                Row.of(null, 0L),
                Row.of(1L, 0L),
                Row.of(1L, 1L),
        });
        expectedOutput.add(new Row[] {
                Row.of(0L, 1L),
                Row.of(null, 1L),
                Row.of(1L, 1L),
                Row.of(1L, 0L),
        });

        expectedOutputNames = new String[]{
                "f0_index",
                "f1_index",
        };

        expectedOutputTypes = new AbstractDataType[]{
                DataTypes.BIGINT(),
                DataTypes.BIGINT()
        };
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test
    public void testUnseenTokenKeepStrategy() {
        errorMessage = "MultiStringIndexer should convert unseen token to (max index + 1) if handleInvalid is set to KEEP strategy.";
        handleInvalid = "KEEP";
        expectedOutput = new HashSet<>();
        expectedOutput.add(new Row[] {
                Row.of("c", 2L, 2L),
                Row.of(null, null, 0L),
                Row.of("b", 0L, 0L),
                Row.of("b", 0L, 1L),
        });
        expectedOutput.add(new Row[] {
                Row.of("c", 2L, 2L),
                Row.of(null, null, 1L),
                Row.of("b", 0L, 1L),
                Row.of("b", 0L, 0L),
        });
        expectedOutput.add(new Row[] {
                Row.of("c", 2L, 2L),
                Row.of(null, null, 0L),
                Row.of("b", 1L, 0L),
                Row.of("b", 1L, 1L),
        });
        expectedOutput.add(new Row[] {
                Row.of("c", 2L, 2L),
                Row.of(null, null, 1L),
                Row.of("b", 1L, 1L),
                Row.of("b", 1L, 0L),
        });
        runAndCheck(tEnv, fields, inputData, inputDataUnseenToken, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test(expected = Exception.class)
    public void testUnseenTokenErrorStrategy() {
        errorMessage = "MultiStringIndexer should throw exception on unseen token if handleInvalid is set to ERROR strategy.";
        handleInvalid = "ERROR";
        runAndCheck(tEnv, fields, inputData, inputDataUnseenToken, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test
    public void testUnseenTokenSkipStrategy() {
        errorMessage = "MultiStringIndexer should convert unseen token to null if handleInvalid is set to SKIP strategy.";
        handleInvalid = "SKIP";
        expectedOutput = new HashSet<>();
        expectedOutput.add(new Row[] {
                Row.of("c", null, null),
                Row.of(null, null, 0L),
                Row.of("b", 0L, 0L),
                Row.of("b", 0L, 1L),
        });
        expectedOutput.add(new Row[] {
                Row.of("c", null, null),
                Row.of(null, null, 1L),
                Row.of("b", 0L, 0L),
                Row.of("b", 0L, 1L),
        });
        runAndCheck(tEnv, fields, inputData, inputDataUnseenToken, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test
    public void testNotSetHandleInvalidStrategy() {
        errorMessage = "MultiStringIndexer should behave like when handleInvalid is set to KEEP strategy if handleInvalid is not set";
        handleInvalid = null;
        expectedOutput = new HashSet<>();
        expectedOutput.add(new Row[] {
                Row.of("c", 2L, 2L),
                Row.of(null, null, 0L),
                Row.of("b", 0L, 0L),
                Row.of("b", 0L, 1L),
        });
        expectedOutput.add(new Row[] {
                Row.of("c", 2L, 2L),
                Row.of(null, null, 1L),
                Row.of("b", 0L, 0L),
                Row.of("b", 0L, 1L),
        });

        runAndCheck(tEnv, fields, inputData, inputDataUnseenToken, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }

    @Test(expected = Exception.class)
    public void testEmptySelectedCol() {
        errorMessage = "MultiStringIndexer should throw Exception when selected columns is empty.";
        selectedCols = new String[0];
        outputCols = new String[0];
        runAndCheck(tEnv, fields, inputData, expectedOutput, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);
    }


    private static void runAndCheck(
            StreamTableEnvironment tEnv,
            DataTypes.Field[] inputType,
            Row[] inputData,
            Set<Row[]> expected,
            String[] expectedOutputNames,
            AbstractDataType[] expectedOutputTypes,
            String[] selectedCols,
            String[] outputCols,
            String[] reservedCols,
            String handleInvalid,
            String errorMessage) {
        runAndCheck(tEnv, inputType, inputData, inputData, expected, expectedOutputNames, expectedOutputTypes, selectedCols, outputCols, reservedCols, handleInvalid, errorMessage);

    }
    private static void runAndCheck(
            StreamTableEnvironment tEnv,
            DataTypes.Field[] inputType,
            Row[] trainData,
            Row[] predictData,
            Set<Row[]> expected,
            String[] expectedOutputNames,
            AbstractDataType[] expectedOutputTypes,
            String[] selectedCols,
            String[] outputCols,
            String[] reservedCols,
            String handleInvalid,
            String errorMessage) {

        Schema.Builder builder = Schema.newBuilder();
        builder.fromFields(expectedOutputNames, expectedOutputTypes);
        Schema expectedSchema = builder.build();

        Table trainTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) trainData);
        Table predictTable = tEnv.fromValues(DataTypes.ROW(inputType), (Object[]) predictData);

        MultiStringIndexer stringIndexer = new MultiStringIndexer();
        if (selectedCols != null)   stringIndexer.setSelectedCols(selectedCols);
        if (outputCols != null)     stringIndexer.setOutputCols(outputCols);
        if (reservedCols != null)   stringIndexer.setReservedCols(reservedCols);
        if (handleInvalid != null)  stringIndexer.setHandleInvalid(handleInvalid);

        Table output = stringIndexer.fit(trainTable).transform(predictTable)[0];
        Assert.assertEquals(errorMessage, expectedSchema, TableUtils.toSchema(output.getResolvedSchema()));

        Object[] actualObjects = IteratorUtils.toArray(output.execute().collect());
        Row[] actual = new Row[actualObjects.length];
        for (int i=0; i<actualObjects.length;i++) {
            actual[i] = (Row) actualObjects[i];
        }
        Map<Object, Integer> actualMap = getFrequencyMap(actual);

        boolean contains = false;
        for (Row[] expectedItem: expected) {
            Map<Object, Integer> expectedMap = getFrequencyMap(expectedItem);
            if (actualMap.equals(expectedMap)) {
                contains = true;
                break;
            }
        }

        Assert.assertTrue(errorMessage, contains);
    }

    private static Map<Object, Integer> getFrequencyMap(Row[] rows) {
        Map<Object, Integer> map = new HashMap<>();
        for (Row row: rows) {
            List<Object> list = toList(row);
            map.put(list, map.getOrDefault(list, 0) + 1);
        }
        return map;
    }

    private static List<Object> toList(Row row) {
        List<Object> list = new ArrayList<>();
        for (int i = 0; i < row.getArity(); i++) {
            list.add(row.getField(i));
        }
        return list;
    }
}
