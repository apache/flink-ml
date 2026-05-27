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

package org.apache.flink.ml.anomalydetection.isolationforest;

import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;

/**
 * Params of {@link IsolationForest}.
 *
 * @param <T> The class of this instance.
 */
public interface IsolationForestParams<T> extends IsolationForestModelParams<T> {
    Param<Integer> MAX_SAMPLES =
            new IntParam(
                    "maxSamples",
                    "The number of samplesData to train and its max value is preferably 256.",
                    256);

    Param<Double> MAX_FEATURES =
            new DoubleParam(
                    "maxFeatures",
                    "The number of features used to train each tree and it is treated as a fraction in the range (0, 1.0].",
                    1.0);

    default int getMaxSamples() {
        return get(MAX_SAMPLES);
    }

    default T setMaxSamples(int value) {
        return set(MAX_SAMPLES, value);
    }

    default double getMaxFeatures() {
        return get(MAX_FEATURES);
    }

    default T setMaxFeatures(double value) {
        return set(MAX_FEATURES, value);
    }
}
