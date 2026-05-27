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

import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasMaxIter;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.common.param.HasRawPredictionCol;
import org.apache.flink.ml.common.param.HasWindows;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;

/**
 * Params of {@link IsolationForestModel}.
 *
 * @param <T> The class of this instance.
 */
public interface IsolationForestModelParams<T>
        extends HasMaxIter<T>,
                HasFeaturesCol<T>,
                HasPredictionCol<T>,
                HasRawPredictionCol<T>,
                HasWindows<T> {
    Param<Integer> NUM_TREES = new IntParam("numTrees", "The max number of ITrees to create.", 100);

    default int getNumTrees() {
        return get(NUM_TREES);
    }

    default T setNumTrees(int value) {
        return set(NUM_TREES, value);
    }
}
