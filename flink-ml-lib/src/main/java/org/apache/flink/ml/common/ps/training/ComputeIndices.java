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

package org.apache.flink.ml.common.ps.training;

import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.linalg.SparseLongDoubleVector;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * An iteration stage that samples a batch of training data and computes the indices needed to
 * compute gradients.
 */
public class ComputeIndices extends ProcessStage<MiniBatchMLSession<LabeledPointWithWeight>> {

    @Override
    public void process(MiniBatchMLSession<LabeledPointWithWeight> context) throws Exception {
        context.readInNextBatchData();
        context.pullIndices = getSortedIndices(context.batchData);
    }

    public static long[] getSortedIndices(List<LabeledPointWithWeight> dataPoints) {
        LongOpenHashSet indices = new LongOpenHashSet();
        for (LabeledPointWithWeight dataPoint : dataPoints) {
            SparseLongDoubleVector feature = (SparseLongDoubleVector) dataPoint.features;
            long[] notZeros = feature.indices;
            for (long index : notZeros) {
                indices.add(index);
            }
        }

        long[] sortedIndices = new long[indices.size()];
        Iterator<Long> iterator = indices.iterator();
        int i = 0;
        while (iterator.hasNext()) {
            sortedIndices[i++] = iterator.next();
        }
        Arrays.sort(sortedIndices);
        return sortedIndices;
    }
}
