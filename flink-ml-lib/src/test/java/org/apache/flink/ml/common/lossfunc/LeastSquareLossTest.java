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

/** Tests {@link LeastSquareLoss}. */
public class LeastSquareLossTest {
    private static final LabeledPointWithWeight dataPoint =
            new LabeledPointWithWeight(Vectors.dense(1.0, 2.0, 3.0), 1.0, 2.0);
    private static final DenseIntDoubleVector coefficient = Vectors.dense(1.0, 1.0, 1.0);
    private static final DenseIntDoubleVector cumGradient = Vectors.dense(0.0, 0.0, 0.0);
    private static final double TOLERANCE = 1e-7;

    @Test
    public void computeLoss() {
        double loss = LeastSquareLoss.INSTANCE.computeLoss(dataPoint, coefficient);
        assertEquals(25.0, loss, TOLERANCE);
    }

    @Test
    public void computeGradient() {
        LeastSquareLoss.INSTANCE.computeGradient(dataPoint, coefficient, cumGradient);
        assertArrayEquals(new double[] {10.0, 20.0, 30.0}, cumGradient.values, TOLERANCE);
        LeastSquareLoss.INSTANCE.computeGradient(dataPoint, coefficient, cumGradient);
        assertArrayEquals(new double[] {20.0, 40.0, 60.0}, cumGradient.values, TOLERANCE);
    }
}
