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

import org.apache.flink.annotation.Internal;
import org.apache.flink.ml.classification.logisticregression.LogisticRegression;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;

/** The loss function for binary logistic loss. See {@link LogisticRegression} for example. */
@Internal
public class BinaryLogisticLoss implements LossFunc {
    public static final BinaryLogisticLoss INSTANCE = new BinaryLogisticLoss();

    private BinaryLogisticLoss() {}

    @Override
    public double computeLoss(LabeledPointWithWeight dataPoint, DenseIntDoubleVector coefficient) {
        double dot = BLAS.dot((IntDoubleVector) dataPoint.features, coefficient);
        double labelScaled = 2 * dataPoint.label - 1;
        return dataPoint.weight * Math.log(1 + Math.exp(-dot * labelScaled));
    }

    @Override
    public void computeGradient(
            LabeledPointWithWeight dataPoint,
            DenseIntDoubleVector coefficient,
            DenseIntDoubleVector cumGradient) {
        IntDoubleVector feature = (IntDoubleVector) dataPoint.features;
        double dot = BLAS.dot(feature, coefficient);
        double labelScaled = 2 * dataPoint.label - 1;
        double multiplier = dataPoint.weight * (-labelScaled / (Math.exp(dot * labelScaled) + 1));
        BLAS.axpy(multiplier, feature, cumGradient, feature.size());
    }

    @Override
    public double computeLoss(double label, double prediction) {
        double labelScaled = 2 * label - 1;
        return Math.log(1 + Math.exp(-prediction * labelScaled));
    }

    @Override
    public double computeGradient(double label, double prediction) {
        double labelScaled = 2 * label - 1;
        return -labelScaled / (Math.exp(prediction * labelScaled) + 1);
    }
}
