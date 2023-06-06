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

import org.apache.flink.ml.linalg.DenseIntDoubleVector;

import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

/** Tests {@link RegularizationUtils}. */
public class RegularizationUtilsTest {
    private static final double learningRate = 0.1;
    private static final double TOLERANCE = 1e-7;
    private static final DenseIntDoubleVector coefficient =
            new DenseIntDoubleVector(new double[] {1.0, -2.0, 0});

    @Test
    public void testRegularization() {
        checkRegularization(0, RandomUtils.nextDouble(0, 1), new double[] {1, -2.0, 0});
        checkRegularization(0.1, 0, new double[] {0.99, -1.98, 0});
        checkRegularization(0.1, 1, new double[] {0.99, -1.99, 0});
        checkRegularization(0.1, 0.1, new double[] {0.99, -1.981, 0});
    }

    private void checkRegularization(double reg, double elasticNet, double[] expectedCoefficient) {
        DenseIntDoubleVector clonedCoefficient = coefficient.clone();
        RegularizationUtils.regularize(clonedCoefficient, reg, elasticNet, learningRate);
        assertArrayEquals(expectedCoefficient, clonedCoefficient.values, TOLERANCE);
    }
}
