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

package org.apache.flink.ml.clustering.kmeans;

import org.apache.flink.ml.param.HasDistanceMeasure;
import org.apache.flink.ml.param.HasFeaturesCol;
import org.apache.flink.ml.param.HasMaxIter;
import org.apache.flink.ml.param.HasPredictionCol;
import org.apache.flink.ml.param.HasSeed;
import org.apache.flink.ml.param.HasTol;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Interface for common params of KMeans and KMeansModel.
 *
 * @param <T> The class type of this instance.
 */
public interface KMeansParams<T>
        extends HasDistanceMeasure<T>,
                HasSeed<T>,
                HasTol<T>,
                HasMaxIter<T>,
                HasFeaturesCol<T>,
                HasPredictionCol<T> {
    Param<Integer> K =
            new IntParam("k", "The number of clusters to create.", 2, ParamValidators.gt(1));

    Param<String> INIT_MODE =
            new StringParam(
                    "initMode",
                    "The initialization algorithm. Supported options: 'random'.",
                    "random",
                    ParamValidators.inArray("random"));

    Param<Integer> INIT_STEPS =
            new IntParam(
                    "initSteps",
                    "The number of steps for k-means initialization mode.",
                    2,
                    ParamValidators.gt(0));

    default int getK() {
        return get(K);
    }

    default T setK(int value) {
        set(K, value);
        return (T) this;
    }

    default String getInitMode() {
        return get(INIT_MODE);
    }

    default T setInitMode(String value) {
        set(INIT_MODE, value);
        return (T) this;
    }

    default int getInitSteps() {
        return get(INIT_STEPS);
    }

    default T setInitSteps(int value) {
        set(INIT_STEPS, value);
        return (T) this;
    }
}
