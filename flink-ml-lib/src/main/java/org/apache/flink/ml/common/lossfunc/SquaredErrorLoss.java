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

/**
 * Squared error loss function defined as (y - pred)^2 where y and pred are label and predictions
 * for the instance respectively.
 */
public class SquaredErrorLoss implements LossFunc {

    public static final SquaredErrorLoss INSTANCE = new SquaredErrorLoss();

    private SquaredErrorLoss() {}

    @Override
    public double loss(double pred, double label) {
        double error = label - pred;
        return error * error;
    }

    @Override
    public double gradient(double pred, double label) {
        return -2. * (label - pred);
    }

    @Override
    public double hessian(double pred, double label) {
        return 2.;
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
