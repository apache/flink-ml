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

import org.apache.flink.ml.common.param.HasDistanceMeasure;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.common.param.HasWindows;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;

/**
 * Params of {@link IsolationForestModel}.
 *
 * @param <T> The class of this instance.
 */
public interface IsolationForestParams<T>
        extends HasDistanceMeasure<T>, HasFeaturesCol<T>, HasPredictionCol<T>, HasWindows<T> {
    Param<Integer> TREES_NUMBER =
            new IntParam("treesNumber", "The max number of trees to create.", 2);

    Param<Integer> ITERS =
            new IntParam("iters", "The max iterations for calculate cluster center.", 2);

    default Integer getTreesNumber() {
        return get(TREES_NUMBER);
    }

    default T setTreesNumber(Integer value) {
        return set(TREES_NUMBER, value);
    }

    default Integer getIters() {
        return get(ITERS);
    }

    default T setIters(Integer value) {
        return set(ITERS, value);
    }
}
