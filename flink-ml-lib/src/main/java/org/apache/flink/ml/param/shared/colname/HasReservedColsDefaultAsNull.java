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

package org.apache.flink.ml.param.shared.colname;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.StringArrayParam;
import org.apache.flink.ml.param.WithParams;

/**
 * An interface for classes with a parameter specifying the names of the columns to be retained in the output table.
 */
public interface HasReservedColsDefaultAsNull<T> extends WithParams<T> {
    Param<String[]> RESERVED_COLS =
            new StringArrayParam(
                    "reservedCols",
                    "Names of the columns to be retained in the output table",
                    null);

    default String[] getReservedCols() {
        return get(RESERVED_COLS);
    }

    default T setReservedCols(String... colNames) {
        return set(RESERVED_COLS, colNames);
    }
}
