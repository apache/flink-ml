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

/** Tests {@link HingeLoss}. */
public class HingeLossTest {
    private static final LabeledPointWithWeight dataPoint1 =
            new LabeledPointWithWeight(Vectors.dense(1.0, -1.0, -1.0), 1.0, 2.0);
    private static final LabeledPointWithWeight dataPoint2 =
            new LabeledPointWithWeight(Vectors.dense(1.0, -1.0, 1.0), 1.0, 2.0);
    private static final DenseIntDoubleVector coefficient = Vectors.dense(1.0, 1.0, 1.0);
    private static final DenseIntDoubleVector cumGradient = Vectors.dense(0.0, 0.0, 0.0);
    private static final double TOLERANCE = 1e-7;

    @Test
    public void computeLoss() {
        double loss = HingeLoss.INSTANCE.computeLoss(dataPoint1, coefficient);
        assertEquals(4.0, loss, TOLERANCE);

        loss = HingeLoss.INSTANCE.computeLoss(dataPoint2, coefficient);
        assertEquals(0.0, loss, TOLERANCE);
    }

    @Test
    public void computeGradient() {
        HingeLoss.INSTANCE.computeGradient(dataPoint1, coefficient, cumGradient);
        assertArrayEquals(new double[] {-2.0, 2.0, 2.0}, cumGradient.values, TOLERANCE);

        HingeLoss.INSTANCE.computeGradient(dataPoint2, coefficient, cumGradient);
        assertArrayEquals(new double[] {-2.0, 2.0, 2.0}, cumGradient.values, TOLERANCE);
    }
}
