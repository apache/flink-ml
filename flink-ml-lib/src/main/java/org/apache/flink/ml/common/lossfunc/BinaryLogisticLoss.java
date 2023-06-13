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
import org.apache.flink.ml.linalg.DenseVector;

/** The loss function for binary logistic loss. See {@link LogisticRegression} for example. */
@Internal
public class BinaryLogisticLoss implements LossFunc {
    public static final BinaryLogisticLoss INSTANCE = new BinaryLogisticLoss();

    private BinaryLogisticLoss() {}

    @Override
    public double computeLoss(LabeledPointWithWeight dataPoint, DenseVector coefficient) {
        double dot = BLAS.dot(dataPoint.getFeatures(), coefficient);
        double labelScaled = 2 * dataPoint.getLabel() - 1;
        return dataPoint.getWeight() * Math.log(1 + Math.exp(-dot * labelScaled));
    }

    @Override
    public void computeGradient(
            LabeledPointWithWeight dataPoint, DenseVector coefficient, DenseVector cumGradient) {
        double dot = BLAS.dot(dataPoint.getFeatures(), coefficient);
        double labelScaled = 2 * dataPoint.getLabel() - 1;
        double multiplier =
                dataPoint.getWeight() * (-labelScaled / (Math.exp(dot * labelScaled) + 1));
        BLAS.axpy(
                multiplier,
                dataPoint.getFeatures(),
                cumGradient,
                (int) dataPoint.getFeatures().size());
    }
}
