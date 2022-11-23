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

import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params of {@link CountVectorizer}.
 *
 * @param <T> The class type of this instance.
 */
public interface CountVectorizerParams<T> extends CountVectorizerModelParams<T> {
    Param<Integer> VOCABULARY_SIZE =
            new IntParam(
                    "vocabularySize",
                    "Max size of the vocabulary. CountVectorizer will build a vocabulary"
                            + "that only considers the top vocabulary size terms ordered by term "
                            + "frequency across the corpus.",
                    1 << 18,
                    ParamValidators.gt(0));

    Param<Double> MIN_DF =
            new DoubleParam(
                    "minDF",
                    "Specifies the minimum number of different documents a term must"
                            + "appear in to be included in the vocabulary. If this is an integer >= 1,"
                            + "this specifies the number of documents the term must appear in;"
                            + "if this is a double in [0,1), then this specifies the fraction of documents.",
                    1.0,
                    ParamValidators.gtEq(0.0));

    Param<Double> MAX_DF =
            new DoubleParam(
                    "maxDF",
                    "Specifies the maximum number of different documents a term could appear "
                            + "in to be included in the vocabulary. A term that appears more than "
                            + "the threshold will be ignored. If this is an integer >= 1, this "
                            + "specifies the maximum number of documents the term could appear in; "
                            + "if this is a double in [0,1), then this specifies the maximum "
                            + "fraction of documents the term could appear in.",
                    (double) Long.MAX_VALUE,
                    ParamValidators.gtEq(0.0));

    default int getVocabularySize() {
        return get(VOCABULARY_SIZE);
    }

    default T setVocabularySize(int value) {
        return set(VOCABULARY_SIZE, value);
    }

    default double getMinDF() {
        return get(MIN_DF);
    }

    default T setMinDF(double value) {
        return set(MIN_DF, value);
    }

    default double getMaxDF() {
        return get(MAX_DF);
    }

    default T setMaxDF(double value) {
        return set(MAX_DF, value);
    }
}
