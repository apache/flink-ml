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

package org.apache.flink.ml.common.datastream;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.catalog.Column;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.types.Row;

/** Utility class for operations related to Table API. */
public class TableUtils {
    // Constructs a RowTypeInfo from the given schema.
    public static RowTypeInfo getRowTypeInfo(ResolvedSchema schema) {
        TypeInformation<?>[] types = new TypeInformation<?>[schema.getColumnCount()];
        String[] names = new String[schema.getColumnCount()];

        for (int i = 0; i < schema.getColumnCount(); i++) {
            Column column = schema.getColumn(i).get();
            types[i] = TypeInformation.of(column.getDataType().getConversionClass());
            names[i] = column.getName();
        }
        return new RowTypeInfo(types, names);
    }

    // Retrieves the TypeInformation of a column by name. Returns null if the name does not exist in
    // the input schema.
    public static TypeInformation<?> getTypeInfoByName(ResolvedSchema schema, String name) {
        for (Column column : schema.getColumns()) {
            if (column.getName().equals(name)) {
                return TypeInformation.of(column.getDataType().getConversionClass());
            }
        }
        return null;
    }

    public static StreamExecutionEnvironment getExecutionEnvironment(StreamTableEnvironment tEnv) {
        Table table = tEnv.fromValues();
        DataStream<Row> dataStream = tEnv.toDataStream(table);
        return dataStream.getExecutionEnvironment();
    }
}
