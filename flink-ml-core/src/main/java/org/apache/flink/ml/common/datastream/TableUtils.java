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
import org.apache.flink.table.runtime.typeutils.ExternalTypeInfo;
import org.apache.flink.table.types.DataType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.types.Row;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Utility class for operations related to Table API. */
public class TableUtils {

    // Logical type roots that may cause wrong type conversion between Table and DataStream.
    private static final Set<LogicalTypeRoot> LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO =
            new HashSet<>();

    static {
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.CHAR);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.VARCHAR);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.BINARY);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.VARBINARY);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.DECIMAL);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.DATE);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.TIME_WITHOUT_TIME_ZONE);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(
                LogicalTypeRoot.TIMESTAMP_WITH_LOCAL_TIME_ZONE);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.INTERVAL_DAY_TIME);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(
                LogicalTypeRoot.TIMESTAMP_WITHOUT_TIME_ZONE);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.ARRAY);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.MAP);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.MULTISET);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.ROW);
        LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.add(LogicalTypeRoot.STRUCTURED_TYPE);
    }

    // Constructs a RowTypeInfo from the given schema. Currently, this function does not support
    // the case when the input contains DataTypes.TIMESTAMP_WITH_TIME_ZONE().
    public static RowTypeInfo getRowTypeInfo(ResolvedSchema schema) {
        TypeInformation<?>[] types = new TypeInformation<?>[schema.getColumnCount()];
        String[] names = new String[schema.getColumnCount()];

        for (int i = 0; i < schema.getColumnCount(); i++) {
            Column column = schema.getColumn(i).get();
            types[i] = getTypeInformationFromDataType(column.getDataType());
            names[i] = column.getName();
        }
        return new RowTypeInfo(types, names);
    }

    // Retrieves the TypeInformation of a column by name. Returns null if the name does not exist in
    // the input schema.
    public static TypeInformation<?> getTypeInfoByName(ResolvedSchema schema, String name) {
        for (Column column : schema.getColumns()) {
            if (column.getName().equals(name)) {
                return getTypeInformationFromDataType(column.getDataType());
            }
        }
        return null;
    }

    public static int[] getColumnIndexes(ResolvedSchema schema, String[] columnNames) {
        Map<String, Integer> nameToIndex = new HashMap<>();
        int[] result = new int[columnNames.length];

        for (int i = 0; i < schema.getColumnCount(); i++) {
            Column column = schema.getColumn(i).get();
            nameToIndex.put(column.getName(), i);
        }

        for (int i = 0; i < columnNames.length; i++) {
            result[i] = nameToIndex.get(columnNames[i]);
        }
        return result;
    }

    public static StreamExecutionEnvironment getExecutionEnvironment(StreamTableEnvironment tEnv) {
        Table table = tEnv.fromValues();
        DataStream<Row> dataStream = tEnv.toDataStream(table);
        return dataStream.getExecutionEnvironment();
    }

    private static TypeInformation<?> getTypeInformationFromDataType(DataType dataType) {
        if (LOGICAL_TYPE_ROOTS_USING_EXTERNAL_TYPE_INFO.contains(
                dataType.getLogicalType().getTypeRoot())) {
            return ExternalTypeInfo.of(dataType);
        }
        return TypeInformation.of(dataType.getConversionClass());
    }
}
