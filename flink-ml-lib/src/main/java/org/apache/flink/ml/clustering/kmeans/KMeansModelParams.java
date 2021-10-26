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

import org.apache.flink.ml.common.param.HasDistanceMeasure;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;

/**
 * Params of KMeansModel.
 *
 * @param <T> The class type of this instance.
 */
public interface KMeansModelParams<T>
        extends HasDistanceMeasure<T>, HasFeaturesCol<T>, HasPredictionCol<T> {

    Param<Integer> K =
            new IntParam("k", "The number of clusters to create.", 2, ParamValidators.gt(1));

    default int getK() {
        return get(K);
    }

    default T setK(int value) {
        set(K, value);
        return (T) this;
    }
}
