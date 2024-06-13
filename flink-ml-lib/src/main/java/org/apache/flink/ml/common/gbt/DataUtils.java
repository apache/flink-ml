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

import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerModel;

import java.util.Arrays;
import java.util.Random;

/** Some data utilities. */
public class DataUtils {

    // Stores 4 values for one histogram bin, i.e., gradient, hessian, weight, and count.
    public static final int BIN_SIZE = 4;

    public static void shuffle(int[] array, Random random) {
        int n = array.length;
        for (int i = 0; i < n; i += 1) {
            int index = i + random.nextInt(n - i);
            int tmp = array[index];
            array[index] = array[i];
            array[i] = tmp;
        }
    }

    public static int[] sample(int[] values, int numSamples, Random random) {
        int n = values.length;
        int[] sampled = new int[numSamples];

        for (int i = 0; i < numSamples; i += 1) {
            int index = i + random.nextInt(n - i);
            sampled[i] = values[index];

            int temp = values[i];
            values[i] = values[index];
            values[index] = temp;
        }
        return sampled;
    }

    /** The mapping computation is from {@link KBinsDiscretizerModel}. */
    public static int findBin(double[] binEdges, double v) {
        int index = Arrays.binarySearch(binEdges, v);
        if (index < 0) {
            // Computes the index to insert.
            index = -index - 1;
            // Puts it in the left bin.
            index--;
        }
        return Math.max(Math.min(index, (binEdges.length - 2)), 0);
    }
}
