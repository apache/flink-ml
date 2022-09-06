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

package org.apache.flink.ml.feature.ngram;

import org.apache.flink.ml.common.param.HasInputCol;
import org.apache.flink.ml.common.param.HasOutputCol;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params of {@link NGram}.
 *
 * @param <T> The class type of this instance.
 */
public interface NGramParams<T> extends HasInputCol<T>, HasOutputCol<T> {
    Param<Integer> N =
            new IntParam("n", "Number of elements per n-gram (>=1).", 2, ParamValidators.gtEq(1));

    default int getN() {
        return get(N);
    }

    default T setN(int value) {
        return set(N, value);
    }
}
