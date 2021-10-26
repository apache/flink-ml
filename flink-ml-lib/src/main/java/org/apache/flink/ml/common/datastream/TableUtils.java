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
import org.apache.flink.table.catalog.Column;
import org.apache.flink.table.catalog.ResolvedSchema;
import org.apache.flink.table.runtime.typeutils.ExternalTypeInfo;

/** Utility class for table-related operations. */
public class TableUtils {
    // Constructs a RowTypeInfo from the given schema.
    public static RowTypeInfo getRowTypeInfo(ResolvedSchema schema) {
        TypeInformation<?>[] types = new TypeInformation<?>[schema.getColumnCount()];
        String[] names = new String[schema.getColumnCount()];

        for (int i = 0; i < schema.getColumnCount(); i++) {
            Column column = schema.getColumn(i).get();
            types[i] = ExternalTypeInfo.of(column.getDataType());
            names[i] = column.getName();
        }
        return new RowTypeInfo(types, names);
    }
}
