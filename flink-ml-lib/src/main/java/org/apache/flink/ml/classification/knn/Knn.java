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

package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the KNN algorithm.
 *
 * <p>See: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm.
 */
public class Knn implements Estimator<Knn, KnnModel>, KnnParams<Knn> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Knn() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public KnnModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        /* Tuple3 : <feature, label, norm square> */
        DataStream<Tuple3<DenseIntDoubleVector, Double, Double>> inputDataWithNorm =
                computeNormSquare(tEnv.toDataStream(inputs[0]));
        DataStream<KnnModelData> modelData = genModelData(inputDataWithNorm);
        KnnModel model = new KnnModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static Knn load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    /**
     * Generates knn model data. For Euclidean distance, distance = sqrt((a - b)^2) = (sqrt(a^2 +
     * b^2 - 2ab)) So it can pre-calculate the L2 norm square of the feature vector, and when
     * calculating the distance with another feature vector, only dot product is calculated. On the
     * other hand, we assemble the feature vectors into a matrix, then it can use blas to accelerate
     * the speed of calculating distances.
     *
     * @param inputDataWithNormSqare Input data with norm square.
     * @return Knn model.
     */
    private static DataStream<KnnModelData> genModelData(
            DataStream<Tuple3<DenseIntDoubleVector, Double, Double>> inputDataWithNormSqare) {
        DataStream<KnnModelData> modelData =
                DataStreamUtils.mapPartition(
                        inputDataWithNormSqare,
                        new RichMapPartitionFunction<
                                Tuple3<DenseIntDoubleVector, Double, Double>, KnnModelData>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple3<DenseIntDoubleVector, Double, Double>>
                                            dataPoints,
                                    Collector<KnnModelData> out) {
                                List<Tuple3<DenseIntDoubleVector, Double, Double>>
                                        bufferedDataPoints = new ArrayList<>();
                                for (Tuple3<DenseIntDoubleVector, Double, Double> dataPoint :
                                        dataPoints) {
                                    bufferedDataPoints.add(dataPoint);
                                }
                                int featureDim = bufferedDataPoints.get(0).f0.size();
                                DenseMatrix packedFeatures =
                                        new DenseMatrix(featureDim, bufferedDataPoints.size());
                                DenseIntDoubleVector normSquares =
                                        new DenseIntDoubleVector(bufferedDataPoints.size());
                                DenseIntDoubleVector labels =
                                        new DenseIntDoubleVector(bufferedDataPoints.size());
                                int offset = 0;
                                for (Tuple3<DenseIntDoubleVector, Double, Double> dataPoint :
                                        bufferedDataPoints) {
                                    System.arraycopy(
                                            dataPoint.f0.values,
                                            0,
                                            packedFeatures.values,
                                            offset * featureDim,
                                            featureDim);
                                    labels.values[offset] = dataPoint.f1;
                                    normSquares.values[offset++] = dataPoint.f2;
                                }
                                out.collect(new KnnModelData(packedFeatures, normSquares, labels));
                            }
                        });
        modelData.getTransformation().setParallelism(1);
        return modelData;
    }

    /**
     * Computes feature norm square.
     *
     * @param inputData Input data.
     * @return Input data with norm square.
     */
    private DataStream<Tuple3<DenseIntDoubleVector, Double, Double>> computeNormSquare(
            DataStream<Row> inputData) {
        return inputData.map(
                new MapFunction<Row, Tuple3<DenseIntDoubleVector, Double, Double>>() {
                    @Override
                    public Tuple3<DenseIntDoubleVector, Double, Double> map(Row value) {
                        Double label = ((Number) value.getField(getLabelCol())).doubleValue();
                        DenseIntDoubleVector feature =
                                ((IntDoubleVector) value.getField(getFeaturesCol())).toDense();
                        return Tuple3.of(feature, label, Math.pow(BLAS.norm2(feature), 2));
                    }
                });
    }
}
