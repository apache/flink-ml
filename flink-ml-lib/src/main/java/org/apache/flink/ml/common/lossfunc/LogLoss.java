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

package org.apache.flink.ml.common.lossfunc;

import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.linalg.DenseVector;

import org.apache.commons.math3.analysis.function.Sigmoid;

/**
 * The loss function for binary log loss.
 *
 * <p>The binary log loss defined as -y * pred + log(1 + exp(pred)) where y is a label in {0, 1} and
 * pred is the predicted logit for the sample point.
 */
public class LogLoss implements LossFunc {

    public static final LogLoss INSTANCE = new LogLoss();
    private final Sigmoid sigmoid = new Sigmoid();

    private LogLoss() {}

    @Override
    public double loss(double pred, double label) {
        return -label * pred + Math.log(1 + Math.exp(pred));
    }

    @Override
    public double gradient(double pred, double label) {
        return sigmoid.value(pred) - label;
    }

    @Override
    public double hessian(double pred, double label) {
        double sig = sigmoid.value(pred);
        return sig * (1 - sig);
    }

    @Override
    public double computeLoss(LabeledPointWithWeight dataPoint, DenseVector coefficient) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void computeGradient(
            LabeledPointWithWeight dataPoint, DenseVector coefficient, DenseVector cumGradient) {
        throw new UnsupportedOperationException();
    }
}
