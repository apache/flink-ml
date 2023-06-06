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
import org.apache.flink.ml.classification.linearsvc.LinearSVC;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;

/**
 * The loss function for hinge loss. See {@link LinearSVC} for example.
 *
 * <p>See https://en.wikipedia.org/wiki/Hinge_loss.
 */
@Internal
public class HingeLoss implements LossFunc {
    public static final HingeLoss INSTANCE = new HingeLoss();

    private HingeLoss() {}

    @Override
    public double computeLoss(LabeledPointWithWeight dataPoint, DenseIntDoubleVector coefficient) {
        double dot = BLAS.dot(dataPoint.getFeatures(), coefficient);
        double labelScaled = 2 * dataPoint.getLabel() - 1;
        return dataPoint.getWeight() * Math.max(0, 1 - labelScaled * dot);
    }

    @Override
    public void computeGradient(
            LabeledPointWithWeight dataPoint,
            DenseIntDoubleVector coefficient,
            DenseIntDoubleVector cumGradient) {
        double dot = BLAS.dot(dataPoint.getFeatures(), coefficient);
        double labelScaled = 2 * dataPoint.getLabel() - 1;
        if (1 - labelScaled * dot > 0) {
            BLAS.axpy(
                    -labelScaled * dataPoint.getWeight(),
                    dataPoint.getFeatures(),
                    cumGradient,
                    dataPoint.getFeatures().size());
        }
    }
}
