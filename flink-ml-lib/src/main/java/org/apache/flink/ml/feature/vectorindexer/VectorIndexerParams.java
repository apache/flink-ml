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

package org.apache.flink.ml.feature.vectorindexer;

import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params of {@link VectorIndexer}.
 *
 * @param <T> The class type of this instance.
 */
public interface VectorIndexerParams<T> extends VectorIndexerModelParams<T> {
    Param<Integer> MAX_CATEGORIES =
            new IntParam(
                    "maxCategories",
                    "Threshold for the number of values a categorical feature can take (>= 2). "
                            + "If a feature is found to have > maxCategories values, then it is declared continuous.",
                    20,
                    ParamValidators.gtEq(2));

    default T setMaxCategories(int value) {
        return set(MAX_CATEGORIES, value);
    }

    default int getMaxCategories() {
        return get(MAX_CATEGORIES);
    }
}
