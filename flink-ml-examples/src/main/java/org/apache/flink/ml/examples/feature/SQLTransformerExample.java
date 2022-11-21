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

package org.apache.flink.ml.examples.feature;

import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.feature.sqltransformer.SQLTransformer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import java.util.Arrays;

/** Simple program that creates a SQLTransformer instance and uses it for feature engineering. */
public class SQLTransformerExample {
    public static void main(String[] args) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        // Generates input data.
        DataStream<Row> inputStream =
                env.fromCollection(
                        Arrays.asList(Row.of(0, 1.0, 3.0), Row.of(2, 2.0, 5.0)),
                        new RowTypeInfo(Types.INT, Types.DOUBLE, Types.DOUBLE));
        Table inputTable = tEnv.fromDataStream(inputStream).as("id", "v1", "v2");

        // Creates a SQLTransformer object and initializes its parameters.
        SQLTransformer sqlTransformer =
                new SQLTransformer()
                        .setStatement("SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__");

        // Uses the SQLTransformer object for feature transformations.
        Table outputTable = sqlTransformer.transform(inputTable)[0];

        // Extracts and displays the results.
        outputTable.execute().print();
    }
}
