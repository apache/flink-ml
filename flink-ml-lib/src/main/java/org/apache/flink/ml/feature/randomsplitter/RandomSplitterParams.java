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

package org.apache.flink.ml.feature.randomsplitter;

import org.apache.flink.ml.common.param.HasSeed;
import org.apache.flink.ml.param.DoubleArrayParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidator;

/**
 * Params of {@link RandomSplitter}.
 *
 * @param <T> The class type of this instance.
 */
public interface RandomSplitterParams<T> extends HasSeed<T> {
    /**
     * Weights should be a non-empty array with all elements greater than zero. The weights will be
     * normalized such that the sum of all elements equals to one.
     */
    Param<Double[]> WEIGHTS =
            new DoubleArrayParam(
                    "weights",
                    "The weights of data splitting.",
                    new Double[] {1.0, 1.0},
                    weightsValidator());

    default T setWeights(Double... value) {
        return set(WEIGHTS, value);
    }

    default Double[] getWeights() {
        return get(WEIGHTS);
    }

    // Checks the weights parameter.
    static ParamValidator<Double[]> weightsValidator() {
        return weights -> {
            if (weights == null) {
                return false;
            }
            for (Double weight : weights) {
                if (weight <= 0.0) {
                    return false;
                }
            }
            return weights.length > 1;
        };
    }
}
