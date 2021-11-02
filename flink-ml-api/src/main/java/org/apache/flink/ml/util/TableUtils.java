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

import org.apache.flink.table.api.Schema;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.types.AbstractDataType;
import org.apache.flink.util.Preconditions;

/**
 * Utility to operator to interact with Table contents, such as rows and columns.
 */
public class TableUtils {
    /**
     * Find the index of <code>targetCol</code> in string array <code>tableCols</code>. It will
     * ignore the case of the tableCols.
     *
     * @param tableCols a string array among which to find the targetCol.
     * @param targetCol the targetCol to find.
     * @return the index of the targetCol, if not found, returns -1.
     */
    public static int findColIndex(String[] tableCols, String targetCol) {
        Preconditions.checkNotNull(targetCol, "targetCol is null!");
        for (int i = 0; i < tableCols.length; i++) {
            if (targetCol.equalsIgnoreCase(tableCols[i])) {
                return i;
            }
        }
        return -1;
    }

    /**
     * convert {@link ResolvedSchema} to corresponding {@link Schema}.
     * @param resolvedSchema a {@link ResolvedSchema}
     * @return the corresponding {@link Schema}
     */
    public static Schema toSchema(ResolvedSchema resolvedSchema) {
        Schema.Builder builder = Schema.newBuilder();
        builder.fromFields(
                resolvedSchema.getColumnNames().toArray(new String[0]),
                resolvedSchema.getColumnDataTypes().toArray(new AbstractDataType[0]));
        return builder.build();
    }
}
