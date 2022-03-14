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

package org.apache.flink.ml.benchmark.datagenerator;

import org.apache.flink.ml.param.LongParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringArrayParam;

/** Interface for generating data as input table arrays. */
public interface InputDataGenerator<T extends InputDataGenerator<T>> extends DataGenerator<T> {
    Param<Long> NUM_VALUES =
            new LongParam(
                    "numValues", "Number of data to be generated.", 10L, ParamValidators.gt(0));

    Param<String[]> COL_NAMES =
            new StringArrayParam(
                    "colNames",
                    "An array of common-separated strings representing field names of data tables.",
                    null);

    default long getNumValues() {
        return get(NUM_VALUES);
    }

    default T setNumValues(long value) {
        return set(NUM_VALUES, value);
    }

    default String[] getColNames() {
        return get(COL_NAMES);
    }

    default T setColNames(String... value) {
        return set(COL_NAMES, value);
    }
}
