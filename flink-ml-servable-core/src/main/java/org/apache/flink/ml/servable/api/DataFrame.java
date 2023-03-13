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

package org.apache.flink.ml.servable.api;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.ml.servable.types.DataType;

import java.util.Iterator;
import java.util.List;

/**
 * A DataFrame consists of several rows, each of which has the same column names and data types.
 *
 * <p>Values in the same column must have the same data type: integer, float, string etc.
 */
@PublicEvolving
public class DataFrame {

    private final List<String> columnNames;
    private final List<DataType> dataTypes;
    private final List<Row> rows;

    /**
     * The given columnNames and dataTypes should be mutable in order for TransformerServable
     * classes to update DataFrame with the serving results.
     */
    public DataFrame(List<String> columnNames, List<DataType> dataTypes, List<Row> rows) {
        int numColumns = columnNames.size();
        if (dataTypes.size() != numColumns) {
            throw new IllegalArgumentException(
                    String.format(
                            "The number of data types %d is different from the number of column names %d.",
                            dataTypes.size(), numColumns));
        }
        for (Row row : rows) {
            if (row.size() != numColumns) {
                throw new IllegalArgumentException(
                        String.format(
                                "The row size %d is different from the number of column names %d.",
                                row.size(), numColumns));
            }
        }

        this.columnNames = columnNames;
        this.dataTypes = dataTypes;
        this.rows = rows;
    }

    /** Returns a list of the names of all the columns in this DataFrame. */
    public List<String> getColumnNames() {
        return columnNames;
    }

    /**
     * Returns the index of the column with the given name.
     *
     * @throws IllegalArgumentException if the column is not present in this table
     */
    public int getIndex(String name) {
        int index = columnNames.indexOf(name);
        if (index == -1) {
            throw new IllegalArgumentException(
                    String.format("Failed to find the column with the name %s.", name));
        }
        return index;
    }

    /**
     * Returns the data type of the column with the given name.
     *
     * @throws IllegalArgumentException if the column is not present in this table
     */
    public DataType getDataType(String name) {
        int index = getIndex(name);
        return dataTypes.get(index);
    }

    /**
     * Adds to this DataFrame a column with the given name, data type, and values.
     *
     * @throws IllegalArgumentException if the number of values is different from the number of
     *     rows.
     */
    public DataFrame addColumn(String columnName, DataType dataType, List<?> values) {
        if (values.size() != rows.size()) {
            throw new RuntimeException(
                    String.format(
                            "The number of values %d is different from the number of rows %d.",
                            values.size(), rows.size()));
        }
        columnNames.add(columnName);
        dataTypes.add(dataType);

        Iterator<?> iter = values.iterator();
        for (Row row : rows) {
            Object value = iter.next();
            row.add(value);
        }
        return this;
    }

    /** Returns all rows of this table. */
    public List<Row> collect() {
        return rows;
    }
}
