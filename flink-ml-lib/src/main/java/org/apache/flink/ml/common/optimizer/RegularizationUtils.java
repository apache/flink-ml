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
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;

/**
 * A utility class for algorithms that need to handle regularization. The regularization term is
 * defined as:
 *
 * <p>elasticNet * reg * norm1(coefficient) + (1 - elasticNet) * (reg/2) * (norm2(coefficient))^2
 *
 * <p>See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html.
 */
@Internal
class RegularizationUtils {

    /**
     * Regularize the model coefficient. The gradient of each dimension could be computed as:
     * {elasticNet * reg * Math.sign(c_i) + (1 - elasticNet) * reg * c_i}. Here c_i is the value of
     * coefficient at i-th dimension.
     *
     * @param coefficient The model coefficient.
     * @param reg The reg param.
     * @param elasticNet The elasticNet param.
     * @param learningRate The learningRate param.
     * @return The loss introduced by regularization.
     */
    public static double regularize(
            DenseIntDoubleVector coefficient,
            final double reg,
            final double elasticNet,
            final double learningRate) {

        if (Double.compare(reg, 0) == 0) {
            return 0;
        } else if (Double.compare(elasticNet, 0) == 0) {
            // Only L2 regularization.
            double loss = reg / 2 * BLAS.norm2(coefficient);
            BLAS.scal(1 - learningRate * reg, coefficient);
            return loss;
        } else if (Double.compare(elasticNet, 1) == 0) {
            // Only L1 regularization.
            double loss = 0;
            double[] coefficientArray = coefficient.values;
            for (int i = 0; i < coefficientArray.length; i++) {
                if (Double.compare(coefficientArray[i], 0) == 0) {
                    continue;
                }
                loss += elasticNet * reg * Math.signum(coefficientArray[i]);
                coefficientArray[i] -=
                        learningRate * elasticNet * reg * Math.signum(coefficientArray[i]);
            }
            return loss;
        } else {
            // Both L1 and L2 are not zero.
            double loss = 0;
            double[] coefficientArray = coefficient.values;
            for (int i = 0; i < coefficientArray.length; i++) {
                loss +=
                        elasticNet * reg * Math.signum(coefficientArray[i])
                                + (1 - elasticNet)
                                        * (reg / 2)
                                        * coefficientArray[i]
                                        * coefficientArray[i];
                coefficientArray[i] -=
                        (learningRate
                                * (elasticNet * reg * Math.signum(coefficientArray[i])
                                        + (1 - elasticNet) * reg * coefficientArray[i]));
            }
            return loss;
        }
    }
}
