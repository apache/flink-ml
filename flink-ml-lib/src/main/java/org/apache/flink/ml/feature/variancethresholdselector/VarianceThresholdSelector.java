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

package org.apache.flink.ml.feature.variancethresholdselector;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * An Estimator which implements the VarianceThresholdSelector algorithm. The algorithm removes all
 * low-variance features. Features with a variance not greater than the threshold will be removed.
 * The default is to keep all features with non-zero variance, i.e. remove the features that have
 * the same value in all samples.
 */
public class VarianceThresholdSelector
        implements Estimator<VarianceThresholdSelector, VarianceThresholdSelectorModel>,
                VarianceThresholdSelectorParams<VarianceThresholdSelector> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public VarianceThresholdSelector() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public VarianceThresholdSelectorModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        final String inputCol = getInputCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Vector> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, Vector>)
                                        value -> ((Vector) value.getField(inputCol)),
                                VectorTypeInfo.INSTANCE);

        DataStream<VarianceThresholdSelectorModelData> modelData =
                DataStreamUtils.aggregate(
                        inputData,
                        new VarianceThresholdSelectorAggregator(getVarianceThreshold()),
                        Types.TUPLE(
                                Types.LONG,
                                DenseVectorTypeInfo.INSTANCE,
                                DenseVectorTypeInfo.INSTANCE),
                        TypeInformation.of(VarianceThresholdSelectorModelData.class));

        VarianceThresholdSelectorModel model =
                new VarianceThresholdSelectorModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    /**
     * A stream operator to compute the variance from feature column of the input bounded data
     * stream.
     */
    private static class VarianceThresholdSelectorAggregator
            implements AggregateFunction<
                    Vector,
                    Tuple3<Long, DenseVector, DenseVector>,
                    VarianceThresholdSelectorModelData> {

        private final double varianceThreshold;

        public VarianceThresholdSelectorAggregator(double varianceThreshold) {
            this.varianceThreshold = varianceThreshold;
        }

        @Override
        public Tuple3<Long, DenseVector, DenseVector> createAccumulator() {
            return Tuple3.of(0L, new DenseVector(new double[0]), new DenseVector(new double[0]));
        }

        @Override
        public Tuple3<Long, DenseVector, DenseVector> add(
                Vector vector, Tuple3<Long, DenseVector, DenseVector> numAndSums) {
            if (numAndSums.f0 == 0) {
                numAndSums.f1 = new DenseVector(vector.size());
                numAndSums.f2 = new DenseVector(vector.size());
            }
            numAndSums.f0 += 1L;
            BLAS.axpy(1.0, vector, numAndSums.f1);
            for (int i = 0; i < vector.size(); i++) {
                numAndSums.f2.values[i] += vector.get(i) * vector.get(i);
            }
            return numAndSums;
        }

        @Override
        public VarianceThresholdSelectorModelData getResult(
                Tuple3<Long, DenseVector, DenseVector> numAndSums) {
            long numRows = numAndSums.f0;
            DenseVector sumVector = numAndSums.f1;
            DenseVector squareSumVector = numAndSums.f2;
            Preconditions.checkState(numRows > 0, "The training set is empty.");

            int[] indices =
                    IntStream.range(0, sumVector.size())
                            .filter(
                                    i ->
                                            squareSumVector.values[i] / numRows
                                                            - (sumVector.values[i] / numRows)
                                                                    * (sumVector.values[i]
                                                                            / numRows)
                                                    > varianceThreshold)
                            .toArray();

            return new VarianceThresholdSelectorModelData(sumVector.size(), indices);
        }

        @Override
        public Tuple3<Long, DenseVector, DenseVector> merge(
                Tuple3<Long, DenseVector, DenseVector> numAndSums1,
                Tuple3<Long, DenseVector, DenseVector> acc) {
            if (numAndSums1.f0 == 0) {
                return acc;
            }

            if (acc.f0 == 0) {
                return numAndSums1;
            }
            acc.f0 += numAndSums1.f0;
            BLAS.axpy(1, numAndSums1.f1, acc.f1);
            BLAS.axpy(1, numAndSums1.f2, acc.f2);
            return acc;
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static VarianceThresholdSelector load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
