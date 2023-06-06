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
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/** Tests {@link BinaryLogisticLoss}. */
public class BinaryLogisticLossTest {
    private static final LabeledPointWithWeight dataPoint =
            new LabeledPointWithWeight(Vectors.dense(1.0, 2.0, 3.0), 1.0, 2.0);
    private static final DenseIntDoubleVector coefficient = Vectors.dense(1.0, 1.0, 1.0);
    private static final DenseIntDoubleVector cumGradient = Vectors.dense(0.0, 0.0, 0.0);
    private static final double TOLERANCE = 1e-7;

    @Test
    public void computeLoss() {
        double loss = BinaryLogisticLoss.INSTANCE.computeLoss(dataPoint, coefficient);
        assertEquals(0.0049513, loss, TOLERANCE);
    }

    @Test
    public void computeGradient() {
        BinaryLogisticLoss.INSTANCE.computeGradient(dataPoint, coefficient, cumGradient);
        assertArrayEquals(
                new double[] {-0.0049452, -0.0098904, -0.0148357}, cumGradient.values, TOLERANCE);
        BinaryLogisticLoss.INSTANCE.computeGradient(dataPoint, coefficient, cumGradient);
        assertArrayEquals(
                new double[] {-0.0098904, -0.0197809, -0.0296714}, cumGradient.values, TOLERANCE);
    }
}
