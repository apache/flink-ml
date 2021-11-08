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

package org.apache.flink.ml.classification.naivebayes;

import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.shared.colname.HasFeatureCols;
import org.apache.flink.ml.param.shared.colname.HasLabelCol;
import org.apache.flink.ml.param.shared.colname.HasPredictionCol;

/**
 * Parameters of naive bayes training process.
 */
public interface NaiveBayesParams<T> extends
        HasFeatureCols<T>,
        HasLabelCol<T>,
        HasPredictionCol<T> {
    Param<Double> SMOOTHING =
            new DoubleParam(
                    "smoothing",
                    "the smoothing factor",
                    0.0,
                    ParamValidators.gtEq(0.0));

    default Double getSmoothing() {
        return get(SMOOTHING);
    }

    default T setSmoothing(Double value) {
        return set(SMOOTHING, value);
    }
}
