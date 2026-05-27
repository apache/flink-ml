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

package org.apache.flink.ml.common.gbt;

import org.junit.Assert;
import org.junit.Test;

/** Test {@link DataUtils}. */
public class DataUtilsTest {
    @Test
    public void testFindBin() {
        double[] binEdges = new double[] {1., 2., 3., 4.};
        for (int i = 0; i < binEdges.length; i += 1) {
            Assert.assertEquals(
                    Math.min(binEdges.length - 2, i), DataUtils.findBin(binEdges, binEdges[i]));
        }
        double[] values = new double[] {.5, 1.5, 2.5, 3.5, 4.5};
        int[] bins = new int[] {0, 0, 1, 2, 2};
        for (int i = 0; i < values.length; i += 1) {
            Assert.assertEquals(bins[i], DataUtils.findBin(binEdges, values[i]));
        }
    }
}
