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

package org.apache.flink.ml.feature.stopwordsremover;

import org.apache.flink.ml.common.param.HasInputCols;
import org.apache.flink.ml.common.param.HasOutputCols;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringArrayParam;
import org.apache.flink.ml.param.StringParam;

import static org.apache.flink.ml.feature.stopwordsremover.StopWordsRemoverUtils.getAvailableLocales;
import static org.apache.flink.ml.feature.stopwordsremover.StopWordsRemoverUtils.getDefaultOrUS;
import static org.apache.flink.ml.feature.stopwordsremover.StopWordsRemoverUtils.loadDefaultStopWords;

/**
 * Params of {@link StopWordsRemover}.
 *
 * @param <T> The class type of this instance.
 */
public interface StopWordsRemoverParams<T> extends HasInputCols<T>, HasOutputCols<T> {

    Param<String[]> STOP_WORDS =
            new StringArrayParam(
                    "stopWords",
                    "The words to be filtered out.",
                    loadDefaultStopWords("english"),
                    ParamValidators.nonEmptyArray());

    Param<Boolean> CASE_SENSITIVE =
            new BooleanParam(
                    "caseSensitive",
                    "Whether to do a case-sensitive comparison over the stop words.",
                    false);

    Param<String> LOCALE =
            new StringParam(
                    "locale",
                    "Locale of the input for case insensitive matching. Ignored when caseSensitive is true.",
                    getDefaultOrUS(),
                    ParamValidators.inArray(getAvailableLocales().toArray(new String[0])));

    default String[] getStopWords() {
        return get(STOP_WORDS);
    }

    default T setStopWords(String[] value) {
        return set(STOP_WORDS, value);
    }

    default boolean getCaseSensitive() {
        return get(CASE_SENSITIVE);
    }

    default T setCaseSensitive(boolean value) {
        return set(CASE_SENSITIVE, value);
    }

    default String getLocale() {
        return get(LOCALE);
    }

    default T setLocale(String value) {
        return set(LOCALE, value);
    }
}
