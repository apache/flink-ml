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

package org.apache.flink.ml.recommendation.fpgrowth;

import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;

/**
 * Params of {@link FPGrowth}.
 *
 * @param <T> The class type of this instance.
 */
public interface FPGrowthParams<T> extends WithParams<T> {

    Param<String> ITEMS_COL =
            new StringParam(
                    "itemsCol", "Item sequence column name.", "items", ParamValidators.notNull());
    Param<String> FIELD_DELIMITER =
            new StringParam(
                    "fieldDelimiter",
                    "Field delimiter of item sequence, default delimiter is ','.",
                    ",",
                    ParamValidators.notNull());
    Param<Double> MIN_LIFT =
            new DoubleParam(
                    "minLift",
                    "Minimal lift level for association rules.",
                    1.0,
                    ParamValidators.gtEq(0));
    Param<Double> MIN_CONFIDENCE =
            new DoubleParam(
                    "minConfidence",
                    "Minimal confidence level for association rules.",
                    0.6,
                    ParamValidators.gtEq(0));
    Param<Double> MIN_SUPPORT =
            new DoubleParam(
                    "minSupport",
                    "Minimal support percent level. The default value of MIN_SUPPORT is 0.02.",
                    0.02);
    Param<Integer> MIN_SUPPORT_COUNT =
            new IntParam(
                    "minSupportCount",
                    "Minimal support count. MIN_ITEM_COUNT has no effect when less than or equal to 0, The default value is -1.",
                    -1);

    Param<Integer> MAX_PATTERN_LENGTH =
            new IntParam(
                    "maxPatternLength", "Max frequent pattern length.", 10, ParamValidators.gt(0));

    default String getItemsCol() {
        return get(ITEMS_COL);
    }

    default T setItemsCol(String value) {
        return set(ITEMS_COL, value);
    }

    default String getFieldDelimiter() {
        return get(FIELD_DELIMITER);
    }

    default T setFieldDelimiter(String value) {
        return set(FIELD_DELIMITER, value);
    }

    default double getMinLift() {
        return get(MIN_LIFT);
    }

    default T setMinLift(Double value) {
        return set(MIN_LIFT, value);
    }

    default Double getMinSupport() {
        return get(MIN_SUPPORT);
    }

    default T setMinSupport(double value) {
        return set(MIN_SUPPORT, value);
    }

    default double getMinConfidence() {
        return get(MIN_CONFIDENCE);
    }

    default T setMinConfidence(Double value) {
        return set(MIN_CONFIDENCE, value);
    }

    default int getMinSupportCount() {
        return get(MIN_SUPPORT_COUNT);
    }

    default T setMinSupportCount(Integer value) {
        return set(MIN_SUPPORT_COUNT, value);
    }

    default int getMaxPatternLength() {
        return get(MAX_PATTERN_LENGTH);
    }

    default T setMaxPatternLength(Integer value) {
        return set(MAX_PATTERN_LENGTH, value);
    }
}
