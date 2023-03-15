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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.feature.sqltransformer.SQLTransformer;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/** Tests {@link SQLTransformer}. */
public class SQLTransformerTest extends AbstractTestBase {
    private static final List<Row> INPUT_DATA =
            Arrays.asList(
                    Row.of(0, 1.0, 3.0),
                    Row.of(1, 2.0, 3.0),
                    Row.of(2, 2.0, 2.0),
                    Row.of(3, 4.0, 2.0));

    private static final List<Row> EXPECTED_NUMERIC_DATA_OUTPUT =
            Arrays.asList(
                    Row.of(0, 1.0, 3.0, 4.0, 3.0),
                    Row.of(1, 2.0, 3.0, 5.0, 6.0),
                    Row.of(2, 2.0, 2.0, 4.0, 4.0),
                    Row.of(3, 4.0, 2.0, 6.0, 8.0));

    private static final List<Row> EXPECTED_BUILT_IN_FUNCTION_OUTPUT =
            Arrays.asList(
                    Row.of(0, 1.0, 3.0, 1.0),
                    Row.of(1, 2.0, 3.0, Math.sqrt(2.0)),
                    Row.of(2, 2.0, 2.0, Math.sqrt(2.0)),
                    Row.of(3, 4.0, 2.0, 2.0));

    private static final List<Row> EXPECTED_GROUP_BY_AGGREGATION_OUTPUT =
            Arrays.asList(Row.of(3.0, 3.0), Row.of(2.0, 6.0));

    private static final List<Row> EXPECTED_WINDOW_AGGREGATION_OUTPUT =
            Collections.singletonList(Row.of(9.0));

    private StreamTableEnvironment tEnv;
    private StreamExecutionEnvironment env;
    private Table inputTable;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
        DataStream<Row> inputStream =
                env.fromCollection(
                        INPUT_DATA, new RowTypeInfo(Types.INT, Types.DOUBLE, Types.DOUBLE));
        inputTable = tEnv.fromDataStream(inputStream).as("id", "v1", "v2");
    }

    @Test
    public void testParam() {
        SQLTransformer sqlTransformer = new SQLTransformer();
        sqlTransformer.setStatement("SELECT * FROM __THIS__");
        assertEquals("SELECT * FROM __THIS__", sqlTransformer.getStatement());
    }

    @Test
    public void testInvalidSQLStatement() {
        SQLTransformer sqlTransformer = new SQLTransformer();

        try {
            sqlTransformer.setStatement("SELECT * FROM __THAT__");
            fail();
        } catch (Exception e) {
            assertEquals(
                    "Parameter statement is given an invalid value SELECT * FROM __THAT__",
                    e.getMessage());
        }
    }

    @Test
    public void testOutputSchema() {
        SQLTransformer sqlTransformer =
                new SQLTransformer()
                        .setStatement("SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__");

        Table outputTable = sqlTransformer.transform(inputTable)[0];

        assertEquals(
                Arrays.asList("id", "v1", "v2", "v3", "v4"),
                outputTable.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testTransformNumericData() {
        SQLTransformer sqlTransformer =
                new SQLTransformer()
                        .setStatement("SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__");

        Table outputTable = sqlTransformer.transform(inputTable)[0];

        verifyOutputResult(outputTable, EXPECTED_NUMERIC_DATA_OUTPUT);
    }

    @Test
    public void testBuiltInFunction() {
        SQLTransformer sqlTransformer =
                new SQLTransformer().setStatement("SELECT *, SQRT(v1) AS v3 FROM __THIS__");

        Table outputTable = sqlTransformer.transform(inputTable)[0];

        verifyOutputResult(outputTable, EXPECTED_BUILT_IN_FUNCTION_OUTPUT);
    }

    @Test
    public void testGroupByAggregation() {
        SQLTransformer sqlTransformer =
                new SQLTransformer()
                        .setStatement("SELECT v2, SUM(v1) AS v3 FROM __THIS__ GROUP BY v2");

        Table outputTable = sqlTransformer.transform(inputTable)[0];

        verifyOutputResult(outputTable, EXPECTED_GROUP_BY_AGGREGATION_OUTPUT);
    }

    @Test
    public void testWindowAggregation() {
        Schema schema =
                Schema.newBuilder()
                        .column("id", DataTypes.INT())
                        .column("v1", DataTypes.DOUBLE())
                        .column("v2", DataTypes.DOUBLE())
                        .columnByExpression("time_ltz", "TO_TIMESTAMP_LTZ(id * 1000, 3)")
                        .watermark("time_ltz", "time_ltz - INTERVAL '5' SECOND")
                        .build();

        DataStream<Row> inputStream =
                env.fromCollection(
                        INPUT_DATA,
                        new RowTypeInfo(
                                new TypeInformation[] {Types.INT, Types.DOUBLE, Types.DOUBLE},
                                new String[] {"id", "v1", "v2"}));
        inputTable = tEnv.fromDataStream(inputStream, schema);

        String statement =
                "SELECT SUM(v1) AS v3 "
                        + "FROM TABLE(TUMBLE(TABLE __THIS__, DESCRIPTOR(time_ltz), INTERVAL '10' MINUTES)) "
                        + "GROUP BY window_start, window_end";

        SQLTransformer sqlTransformer = new SQLTransformer().setStatement(statement);

        Table outputTable = sqlTransformer.transform(inputTable)[0];

        verifyOutputResult(outputTable, EXPECTED_WINDOW_AGGREGATION_OUTPUT);
    }

    @Test
    public void testSaveLoadAndTransform() throws Exception {
        SQLTransformer sqlTransformer =
                new SQLTransformer()
                        .setStatement("SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__");

        SQLTransformer loadedSQLTransformer =
                TestUtils.saveAndReload(
                        tEnv,
                        sqlTransformer,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        SQLTransformer::load);

        Table outputTable = loadedSQLTransformer.transform(inputTable)[0];

        verifyOutputResult(outputTable, EXPECTED_NUMERIC_DATA_OUTPUT);
    }

    private static void verifyOutputResult(Table outputTable, List<Row> expectedOutput) {
        List<Row> actualOutput = IteratorUtils.toList(outputTable.execute().collect());
        assertEquals(new HashSet<>(expectedOutput), new HashSet<>(actualOutput));
    }
}
