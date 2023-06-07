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

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.feature.LabeledPointWithWeight;
import org.apache.flink.ml.common.lossfunc.LossFunc;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.SparseLongDoubleVector;
import org.apache.flink.ml.linalg.Vectors;

import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;

import java.io.IOException;
import java.util.List;

/** An iteration stage that uses the pulled model values and batch data to compute the gradients. */
public class ComputeGradients extends ProcessStage<MiniBatchMLSession<LabeledPointWithWeight>> {
    private final LossFunc lossFunc;

    public ComputeGradients(LossFunc lossFunc) {
        this.lossFunc = lossFunc;
    }

    @Override
    public void process(MiniBatchMLSession<LabeledPointWithWeight> session) throws IOException {
        long[] indices = ComputeIndices.getSortedIndices(session.batchData);
        double[] modelValues = session.pulledValues;
        double[] gradients = computeGradient(session.batchData, Tuple2.of(indices, modelValues));

        session.pushIndices = indices;
        session.pushValues = gradients;
    }

    private double[] computeGradient(
            List<LabeledPointWithWeight> batchData, Tuple2<long[], double[]> modelData) {
        long[] modelIndices = modelData.f0;
        double[] modelValues = modelData.f1;
        Long2DoubleOpenHashMap modelInMap = new Long2DoubleOpenHashMap(modelIndices.length);
        for (int i = 0; i < modelIndices.length; i++) {
            modelInMap.put(modelIndices[i], modelValues[i]);
        }
        Long2DoubleOpenHashMap cumGradients = new Long2DoubleOpenHashMap(modelIndices.length);

        for (LabeledPointWithWeight dataPoint : batchData) {
            SparseLongDoubleVector feature = (SparseLongDoubleVector) dataPoint.features;
            double dot = dot(feature, modelInMap);
            double multiplier = lossFunc.computeGradient(dataPoint.label, dot) * dataPoint.weight;

            long[] featureIndices = feature.indices;
            double[] featureValues = feature.values;
            double z;
            for (int i = 0; i < featureIndices.length; i++) {
                long currentIndex = featureIndices[i];
                z = featureValues[i] * multiplier + cumGradients.getOrDefault(currentIndex, 0.);
                cumGradients.put(currentIndex, z);
            }
        }
        double[] cumGradientValues = new double[modelIndices.length];
        for (int i = 0; i < modelIndices.length; i++) {
            cumGradientValues[i] = cumGradients.get(modelIndices[i]);
        }
        BLAS.scal(1.0 / batchData.size(), Vectors.dense(cumGradientValues));
        return cumGradientValues;
    }

    private static double dot(SparseLongDoubleVector feature, Long2DoubleOpenHashMap coefficient) {
        double dot = 0;
        for (int i = 0; i < feature.indices.length; i++) {
            dot += feature.values[i] * coefficient.get(feature.indices[i]);
        }
        return dot;
    }
}
