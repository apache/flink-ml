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

package org.apache.flink.ml.feature.lsh;

import org.apache.flink.ml.common.param.HasInputCol;
import org.apache.flink.ml.common.param.HasOutputCol;
import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params for {@link LSH}.
 *
 * @param <T> The class type of this instance.
 */
public interface LSHParams<T> extends HasInputCol<T>, HasOutputCol<T>, HasSeed<T> {

    /**
     * Param for the number of hash tables used in LSH OR-amplification.
     *
     * <p>OR-amplification can be used to reduce the false negative rate. Higher values of this
     * param lead to a reduced false negative rate, at the expense of added computational
     * complexity.
     */
    Param<Integer> NUM_HASH_TABLES =
            new IntParam("numHashTables", "Number of hash tables.", 1, ParamValidators.gtEq(1.));

    default int getNumHashTables() {
        return get(NUM_HASH_TABLES);
    }

    default T setNumHashTables(Integer value) {
        return set(NUM_HASH_TABLES, value);
    }

    /**
     * Param for the number of hash functions per hash table used in LSH AND-amplification.
     *
     * <p>AND-amplification can be used to reduce the false positive rate. Higher values of this
     * param lead to a reduced false positive rate, at the expense of added computational
     * complexity.
     */
    Param<Integer> NUM_HASH_FUNCTIONS_PER_TABLE =
            new IntParam(
                    "numHashFunctionsPerTable",
                    "Number of hash functions per table.",
                    1,
                    ParamValidators.gtEq(1.));

    default int getNumHashFunctionsPerTable() {
        return get(NUM_HASH_FUNCTIONS_PER_TABLE);
    }

    default T setNumHashFunctionsPerTable(Integer value) {
        return set(NUM_HASH_FUNCTIONS_PER_TABLE, value);
    }
}
