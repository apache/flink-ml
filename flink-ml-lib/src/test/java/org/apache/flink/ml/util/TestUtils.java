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

package org.apache.flink.ml.util;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;

import org.apache.commons.lang3.ArrayUtils;

import java.lang.reflect.Method;

/** Utility methods for unit tests. */
public class TestUtils {
    /**
     * Saves a stage to filesystem and reloads it by invoking the static load() method of the given
     * stage.
     */
    public static <T extends Stage<T>> T saveAndReload(
            StreamTableEnvironment tEnv, T stage, String path) throws Exception {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        stage.save(path);
        try {
            env.execute();
        } catch (RuntimeException e) {
            if (!e.getMessage()
                    .equals("No operators defined in streaming topology. Cannot execute.")) {
                throw e;
            }
        }

        Method method =
                stage.getClass().getMethod("load", StreamTableEnvironment.class, String.class);
        return (T) method.invoke(null, tEnv, path);
    }

    /**
     * Converts data types in the table to sparse types and integer types.
     *
     * <ul>
     *   <li>If a column in the table is of DenseVector type, converts it to SparseVector.
     *   <li>If a column in the table is of Double type, converts it to integer.
     * </ul>
     */
    public static Table convertDataTypesToSparseInt(StreamTableEnvironment tEnv, Table table) {
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(table.getResolvedSchema());
        TypeInformation<?>[] fieldTypes = inputTypeInfo.getFieldTypes();
        for (int i = 0; i < fieldTypes.length; i++) {
            if (fieldTypes[i].getTypeClass().equals(DenseVector.class)) {
                fieldTypes[i] = SparseVectorTypeInfo.INSTANCE;
            } else if (fieldTypes[i].getTypeClass().equals(Double.class)) {
                fieldTypes[i] = Types.INT;
            }
        }

        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(fieldTypes),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames()));
        DataStream<Row> dataStream = tEnv.toDataStream(table);
        dataStream =
                dataStream.map(
                        new MapFunction<Row, Row>() {
                            @Override
                            public Row map(Row row) {
                                int arity = row.getArity();
                                for (int i = 0; i < arity; i++) {
                                    Object obj = row.getField(i);
                                    if (obj instanceof Vector) {
                                        row.setField(i, ((Vector) obj).toSparse());
                                    } else if (obj instanceof Number) {
                                        row.setField(i, ((Number) obj).intValue());
                                    }
                                }
                                return row;
                            }
                        },
                        outputTypeInfo);
        return tEnv.fromDataStream(dataStream);
    }

    /** Gets the types of data in each column of the input table. */
    public static Class<?>[] getColumnDataTypes(Table table) {
        return table.getResolvedSchema().getColumnDataTypes().stream()
                .map(DataType::getConversionClass)
                .toArray(Class<?>[]::new);
    }

    /** Note: this comparator imposes orderings that are inconsistent with equals. */
    public static int compare(Vector first, Vector second) {
        if (first.size() != second.size()) {
            return Integer.compare(first.size(), second.size());
        } else {
            for (int i = 0; i < first.size(); i++) {
                int cmp = Double.compare(first.get(i), second.get(i));
                if (cmp != 0) {
                    return cmp;
                }
            }
        }
        return 0;
    }
}
