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

package org.apache.flink.ml.common.optimizer;

import org.apache.flink.annotation.Internal;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.streaming.api.datastream.DataStream;

/**
 * An optimizer is a function to modify the weight of a machine learning model, which aims to find
 * the optimal parameter configuration for a machine learning model. Examples of optimizers could be
 * stochastic gradient descent (SGD), L-BFGS, etc.
 */
@Internal
public interface Optimizer {
    /**
     * Optimizes the given loss function using the initial model data and the bounded training data.
     *
     * @param initModelData The initial model data.
     * @param trainData The training data.
     * @param lossFunc The loss function to optimize.
     * @return The fitted model data.
     */
    DataStream<DenseIntDoubleVector> optimize(
            DataStream<DenseIntDoubleVector> initModelData,
            DataStream<LabeledPointWithWeight> trainData,
            LossFunc lossFunc);
}
