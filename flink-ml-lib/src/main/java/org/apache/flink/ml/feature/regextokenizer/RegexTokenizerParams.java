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

package org.apache.flink.ml.feature.regextokenizer;

import org.apache.flink.ml.common.param.HasInputCol;
import org.apache.flink.ml.common.param.HasOutputCol;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link RegexTokenizer}.
 *
 * @param <T> The class type of this instance.
 */
public interface RegexTokenizerParams<T> extends HasInputCol<T>, HasOutputCol<T> {
    IntParam MIN_TOKEN_LENGTH =
            new IntParam("minTokenLength", "Minimum token length", 1, ParamValidators.gtEq(0));

    BooleanParam GAPS = new BooleanParam("gaps", "Set regex to match gaps or tokens", true);

    StringParam PATTERN = new StringParam("pattern", "Regex pattern used for tokenizing", "\\s+");

    BooleanParam TO_LOWERCASE =
            new BooleanParam(
                    "toLowercase",
                    "Whether to convert all characters to lowercase before tokenizing",
                    true);

    default T setMinTokenLength(int value) {
        return set(MIN_TOKEN_LENGTH, value);
    }

    default int getMinTokenLength() {
        return get(MIN_TOKEN_LENGTH);
    }

    default T setGaps(boolean value) {
        return set(GAPS, value);
    }

    default Boolean getGaps() {
        return get(GAPS);
    }

    default T setPattern(String pattern) {
        return set(PATTERN, pattern);
    }

    default String getPattern() {
        return get(PATTERN);
    }

    default T setToLowercase(boolean toLowercase) {
        return set(TO_LOWERCASE, toLowercase);
    }

    default Boolean getToLowercase() {
        return get(TO_LOWERCASE);
    }
}
