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

package org.apache.flink.ml.feature.sqltransformer;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.common.datastream.EndOfStreamWindows;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.TableException;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.types.RowKind;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * SQLTransformer implements the transformations that are defined by SQL statement.
 *
 * <p>Currently we only support SQL syntax like `SELECT ... FROM __THIS__ ...` where `__THIS__`
 * represents the input table and cannot be modified.
 *
 * <p>The select clause specifies the fields, constants, and expressions to display in the output.
 * Except the cases described in the note section below, it can be any select clause that Flink SQL
 * supports. Users can also use Flink SQL built-in function and UDFs to operate on these selected
 * columns.
 *
 * <p>For example, SQLTransformer supports statements like:
 *
 * <ul>
 *   <li>`SELECT a, a + b AS a_b FROM __THIS__`
 *   <li>`SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5`
 *   <li>`SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b`
 * </ul>
 *
 * <p>Note: This operator only generates append-only/insert-only table as its output. If the output
 * table could possibly contain retract messages(e.g. perform `SELECT ... FROM __THIS__ GROUP BY
 * ...` operation on a table in streaming mode), this operator would aggregate all changelogs and
 * only output the final state.
 */
public class SQLTransformer
        implements Transformer<SQLTransformer>, SQLTransformerParams<SQLTransformer> {
    static final String TABLE_IDENTIFIER = "__THIS__";

    private static final String INSERT_ONLY_EXCEPTION_PATTERN =
            "^.* doesn't support consuming .* changes which is produced by node .*$";

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public SQLTransformer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        String statement = getStatement().replace(TABLE_IDENTIFIER, inputs[0].toString());

        Table outputTable = tEnv.sqlQuery(statement);

        if (!isInsertOnlyTable(tEnv, outputTable)) {
            Schema schema =
                    Schema.newBuilder().fromResolvedSchema(outputTable.getResolvedSchema()).build();
            DataStream<Row> outputStream = tEnv.toChangelogStream(outputTable, schema);

            outputStream =
                    outputStream
                            .windowAll(EndOfStreamWindows.get())
                            .aggregate(
                                    new ChangeLogStreamToDataStreamFunction(),
                                    Types.LIST(outputStream.getType()),
                                    Types.LIST(outputStream.getType()))
                            .flatMap(new FlattenListFunction<>(), outputStream.getType());

            outputTable = tEnv.fromDataStream(outputStream, schema);
        }

        return new Table[] {outputTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static SQLTransformer load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private boolean isInsertOnlyTable(StreamTableEnvironment tEnv, Table table) {
        try {
            tEnv.toDataStream(table);
            return true;
        } catch (Exception e) {
            if (e instanceof TableException
                    && e.getMessage() != null
                    && e.getMessage().matches(INSERT_ONLY_EXCEPTION_PATTERN)) {
                return false;
            }
            throw e;
        }
    }

    /**
     * A function that converts a bounded changelog stream to an insert-only datastream. It
     * aggregates all records in a bounded changelog stream and outputs each record in the
     * aggregation result. Records are output according to their last modification time.
     */
    private static class ChangeLogStreamToDataStreamFunction
            implements AggregateFunction<Row, List<Row>, List<Row>> {
        @Override
        public List<Row> createAccumulator() {
            return new ArrayList<>();
        }

        @Override
        public List<Row> add(Row value, List<Row> accumulator) {
            switch (value.getKind()) {
                case INSERT:
                    accumulator.add(value);
                    break;
                case UPDATE_AFTER:
                    value.setKind(RowKind.INSERT);
                    accumulator.add(value);
                    break;
                case UPDATE_BEFORE:
                case DELETE:
                    value.setKind(RowKind.INSERT);
                    accumulator.remove(value);
                    break;
                default:
                    throw new UnsupportedOperationException();
            }
            return accumulator;
        }

        @Override
        public List<Row> getResult(List<Row> accumulator) {
            return accumulator;
        }

        @Override
        public List<Row> merge(List<Row> a, List<Row> b) {
            a.addAll(b);
            return a;
        }
    }

    private static class FlattenListFunction<T> implements FlatMapFunction<List<T>, T> {
        @Override
        public void flatMap(List<T> values, Collector<T> out) throws Exception {
            for (T value : values) {
                out.collect(value);
            }
        }
    }
}
