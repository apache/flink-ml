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

package org.apache.flink.ml.clustering.kmeans;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.core.Estimator;
import org.apache.flink.ml.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.util.Collector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * An Estimator which implements the k-means clustering algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/K-means_clustering.
 */
public class KMeans implements Estimator<KMeans, KMeansModel>, KMeansParams<KMeans> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    @Override
    public KMeansModel fit(Table... inputs) {
        return null;
    }

    @Override
    public KMeansModel fitDataSet(DataSet... inputs) throws Exception {
        DataSet<DenseVector> points = inputs[0];

        DataSet<Tuple2<Integer, DenseVector>> initCentroids = initRandom(points, getK(), getSeed());

        IterativeDataSet<Tuple2<Integer, DenseVector>> loopCentroids =
                initCentroids.iterate(getMaxIter());

        DataSet<Tuple2<Integer, DenseVector>> newCentroids =
                points.map(new SelectNearestCentroid())
                        .withBroadcastSet(loopCentroids, "centroids")
                        .map(new CountAppender())
                        .groupBy(0)
                        .reduce(new CentroidAccumulator())
                        .map(new CentroidAverager());

        DataSet<Tuple2<Integer, DenseVector>> finalCentroids =
                loopCentroids.closeWith(newCentroids);

        return new KMeansModel(finalCentroids);
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static KMeans load(String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getUserDefinedParamMap() {
        return paramMap;
    }

    private static class CentroidAverager
            implements MapFunction<
                    Tuple3<Integer, DenseVector, Long>, Tuple2<Integer, DenseVector>> {

        @Override
        public Tuple2<Integer, DenseVector> map(Tuple3<Integer, DenseVector, Long> value) {
            for (int i = 0; i < value.f1.size(); i++) {
                value.f1.values[i] /= value.f2;
            }
            return Tuple2.of(value.f0, value.f1);
        }
    }

    private static class CentroidAccumulator
            implements ReduceFunction<Tuple3<Integer, DenseVector, Long>> {
        @Override
        public Tuple3<Integer, DenseVector, Long> reduce(
                Tuple3<Integer, DenseVector, Long> v1, Tuple3<Integer, DenseVector, Long> v2)
                throws Exception {
            for (int i = 0; i < v1.f1.size(); i++) {
                v1.f1.values[i] += v2.f1.values[i];
            }
            return new Tuple3<>(v1.f0, v1.f1, v1.f2 + v2.f2);
        }
    }

    private static class CountAppender
            implements MapFunction<
                    Tuple2<Integer, DenseVector>, Tuple3<Integer, DenseVector, Long>> {
        @Override
        public Tuple3<Integer, DenseVector, Long> map(Tuple2<Integer, DenseVector> value) {
            return Tuple3.of(value.f0, value.f1, 1L);
        }
    }

    /** Select the nearest centroid for every point. */
    public static class SelectNearestCentroid
            extends RichMapFunction<DenseVector, Tuple2<Integer, DenseVector>> {
        private List<Tuple2<Integer, DenseVector>> centroids;

        @Override
        public void open(Configuration parameters) throws Exception {
            centroids = getRuntimeContext().getBroadcastVariable("centroids");
        }

        @Override
        public Tuple2<Integer, DenseVector> map(DenseVector point) throws Exception {
            double minDistance = Double.MAX_VALUE;
            int closestCentroidId = -1;
            EuclideanDistanceMeasure measure = new EuclideanDistanceMeasure();

            for (Tuple2<Integer, DenseVector> centroid : centroids) {
                double distance = measure.distance(centroid.f1, point);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCentroidId = centroid.f0;
                }
            }
            return Tuple2.of(closestCentroidId, point);
        }
    }

    private static DataSet<Tuple2<Integer, DenseVector>> initRandom(
            DataSet<DenseVector> data, int k, long seed) {
        return data.mapPartition(
                        new MapPartitionFunction<DenseVector, Tuple2<Integer, DenseVector>>() {
                            @Override
                            public void mapPartition(
                                    Iterable<DenseVector> iterable,
                                    Collector<Tuple2<Integer, DenseVector>> out)
                                    throws Exception {
                                List<DenseVector> vectors = new ArrayList<>();
                                for (DenseVector vector : iterable) {
                                    vectors.add(vector);
                                }
                                Collections.shuffle(vectors, new Random(seed));
                                for (int i = 0; i < k; i++) {
                                    out.collect(Tuple2.of(i, vectors.get(i)));
                                }
                            }
                        })
                .setParallelism(1);
    }
}
