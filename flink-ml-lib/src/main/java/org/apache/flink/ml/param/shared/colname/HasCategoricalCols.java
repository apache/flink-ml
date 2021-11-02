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
 * Param: columns whose type are string or boolean.
 */
public interface HasCategoricalCols<T> extends WithParams<T> {
    Param<String[]> CATEGORICAL_COLS =
            new StringArrayParam(
                    "categoricalCols",
                    "Names of the categorical columns used for training in the input table",
                    null);

    default String[] getCategoricalCols() {
        return get(CATEGORICAL_COLS);
    }

    default T setCategoricalCols(String... colNames) {
        return set(CATEGORICAL_COLS, colNames);
    }
}
