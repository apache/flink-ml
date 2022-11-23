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

package org.apache.flink.ml.feature.countvectorizer;

import org.apache.flink.ml.common.param.HasInputCol;
import org.apache.flink.ml.common.param.HasOutputCol;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params for {@link CountVectorizerModel}.
 *
 * @param <T> The class type of this instance.
 */
public interface CountVectorizerModelParams<T> extends HasInputCol<T>, HasOutputCol<T> {
    Param<Double> MIN_TF =
            new DoubleParam(
                    "minTF",
                    "Filter to ignore rare words in a document. For each document,"
                            + "terms with frequency/count less than the given threshold are ignored. "
                            + "If this is an integer >= 1, then this specifies a count (of times "
                            + "the term must appear in the document); if this is a double in [0,1), "
                            + "then this specifies a fraction (out of the document's token count).",
                    1.0,
                    ParamValidators.gtEq(0.0));

    Param<Boolean> BINARY =
            new BooleanParam(
                    "binary",
                    "Binary toggle to control the output vector values. If True, all nonzero "
                            + "counts (after minTF filter applied) are set to 1.0.",
                    false);

    default double getMinTF() {
        return get(MIN_TF);
    }

    default T setMinTF(double value) {
        return set(MIN_TF, value);
    }

    default boolean getBinary() {
        return get(BINARY);
    }

    default T setBinary(boolean value) {
        return set(BINARY, value);
    }
}
