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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.ml.classification.gbtclassifier.GBTClassifierModel;
import org.apache.flink.ml.common.param.HasCategoricalCols;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasLabelCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.StringArrayParam;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressorModel;

/**
 * Params of {@link GBTClassifierModel} and {@link GBTRegressorModel}.
 *
 * <p>If the input features come from 1 column of vector type, `featuresCol` should be used, and all
 * features are treated as continuous features. Otherwise, `inputCols` should be used for multiple
 * columns. Columns whose names specified in `categoricalCols` are treated as categorical features,
 * while others are continuous features.
 *
 * <p>NOTE: `inputCols` and `featuresCol` are in conflict with each other, so they should not be set
 * at the same time. In addition, `inputCols` has a higher precedence than `featuresCol`, that is,
 * `featuresCol` is ignored when `inputCols` is not `null`.
 *
 * @param <T> The class type of this instance.
 */
public interface GBTModelParams<T>
        extends HasFeaturesCol<T>, HasLabelCol<T>, HasCategoricalCols<T>, HasPredictionCol<T> {

    Param<String[]> INPUT_COLS = new StringArrayParam("inputCols", "Input column names.", null);

    default String[] getInputCols() {
        return get(INPUT_COLS);
    }

    default T setInputCols(String... value) {
        return set(INPUT_COLS, value);
    }
}
