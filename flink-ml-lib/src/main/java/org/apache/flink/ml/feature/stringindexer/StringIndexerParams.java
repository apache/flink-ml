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

package org.apache.flink.ml.feature.stringindexer;

import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params of {@link StringIndexer}.
 *
 * @param <T> The class type of this instance.
 */
public interface StringIndexerParams<T> extends StringIndexerModelParams<T> {
    String ARBITRARY_ORDER = "arbitrary";
    String FREQUENCY_DESC_ORDER = "frequencyDesc";
    String FREQUENCY_ASC_ORDER = "frequencyAsc";
    String ALPHABET_DESC_ORDER = "alphabetDesc";
    String ALPHABET_ASC_ORDER = "alphabetAsc";

    /**
     * Supported options to decide the order of strings in each column are listed as follows. (The
     * first label after ordering is assigned an index of 0).
     *
     * <ul>
     *   <li>arbitrary: the order of strings is arbitrary and depends on each execution.
     *   <li>frequencyDesc: descending order by the frequency.
     *   <li>frequencyAsc: ascending order by the frequency.
     *   <li>alphabetDesc: descending alphabetical order.
     *   <li>alphabetAsc: descending alphabetical order.
     * </ul>
     */
    Param<String> STRING_ORDER_TYPE =
            new StringParam(
                    "stringOrderType",
                    "How to order strings of each column.",
                    ARBITRARY_ORDER,
                    ParamValidators.inArray(
                            ARBITRARY_ORDER,
                            FREQUENCY_DESC_ORDER,
                            FREQUENCY_ASC_ORDER,
                            ALPHABET_DESC_ORDER,
                            ALPHABET_ASC_ORDER));

    Param<Integer> MAX_INDEX_NUM =
            new IntParam(
                    "maxIndexNum",
                    "The max number of indices for each column. It only works when "
                            + "'stringOrderType' is set as 'frequencyDesc'.",
                    Integer.MAX_VALUE,
                    ParamValidators.gt(1));

    default String getStringOrderType() {
        return get(STRING_ORDER_TYPE);
    }

    default T setStringOrderType(String value) {
        return set(STRING_ORDER_TYPE, value);
    }

    default int getMaxIndexNum() {
        return get(MAX_INDEX_NUM);
    }

    default T setMaxIndexNum(int value) {
        return set(MAX_INDEX_NUM, value);
    }
}
