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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.api.core.Estimator;
import org.apache.flink.ml.common.EndOfStreamWindows;
import org.apache.flink.ml.common.MapPartitionFunctionWrapper;
import org.apache.flink.ml.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import org.apache.commons.collections.IteratorUtils;

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
    public KMeansModel fit(Table... inputs) throws Exception {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> values = tEnv.toDataStream(inputs[0]);

        DataStream<DenseVector> points = values.map(row -> (DenseVector) row.getField("f0"));

        DataStream<Tuple2<Integer, DenseVector>> initCentroids =
                initRandom(points, getK(), getSeed());

        IterationBody body =
                new IterationBody() {
                    @Override
                    public IterationBodyResult process(
                            DataStreamList variableStreams, DataStreamList dataStreams) {
                        DataStream<Tuple2<Integer, DenseVector>> centroids = variableStreams.get(0);
                        DataStream<DenseVector> points = dataStreams.get(0);

                        DataStream<Integer> terminationCriteria =
                                points.flatMap(new RoundBasedTerminationCriteria(10));

                        TypeInformation<Tuple2<Integer, DenseVector>> typeInfo =
                                new TupleTypeInfo<Tuple2<Integer, DenseVector>>(
                                        BasicTypeInfo.INT_TYPE_INFO,
                                        TypeInformation.of(DenseVector.class));

                        DataStream<Tuple2<Integer, DenseVector>> newCentroids =
                                points.connect(centroids.broadcast())
                                        .transform(
                                                "SelectNearestCentroid",
                                                typeInfo,
                                                new SelectNearestCentroidOperator())
                                        .map(new CountAppender())
                                        .keyBy(t -> t.f0)
                                        .window(EndOfStreamWindows.get())
                                        .reduce(new CentroidAccumulator())
                                        .map(new CentroidAverager());

                        return new IterationBodyResult(
                                DataStreamList.of(newCentroids),
                                variableStreams,
                                terminationCriteria);
                    }
                };

        DataStreamList output =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(initCentroids),
                        ReplayableDataStreamList.replay(points),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(IterationConfig.OperatorLifeCycle.PER_ROUND)
                                .build(),
                        body);

        DataStream<Tuple2<Integer, DenseVector>> finalCentroids = output.get(0);

        List<Tuple2<Integer, DenseVector>> result =
                IteratorUtils.toList(finalCentroids.executeAndCollect());

        System.out.println("result " + result.size());

        return null;

        //        DataStreamList outputs =
        //                Iterations.iterateBoundedStreamsUntilTermination(
        //                        DataStreamList.of(variableSource),
        //                        ReplayableDataStreamList.notReplay(constSource),
        //                        IterationConfig.newBuilder().build(),
        //                        (variableStreams, dataStreams) -> {
        //                            SingleOutputStreamOperator<Integer> reducer =
        //                                    variableStreams
        //                                            .<Integer>get(0)
        //                                            .connect(dataStreams.<Integer>get(0))
        //                                            .process(
        //                                                    new
        // TwoInputReduceAllRoundProcessFunction(
        //                                                            sync, maxRound * 10));
        //                            return new IterationBodyResult(
        //                                    DataStreamList.of(
        //                                            reducer.map(x ->
        // x).setParallelism(numSources)),
        //                                    DataStreamList.of(
        //                                            reducer.getSideOutput(
        //                                                    new OutputTag<OutputRecord<Integer>>(
        //                                                            "output") {})),
        //                                    reducer.flatMap(new
        // RoundBasedTerminationCriteria(maxRound)));
        //                        });
    }

    @Override
    public KMeansModel fitDataSet(DataSet... inputs) throws Exception {
        return null;
        //        DataSet<DenseVector> points = inputs[0];
        //
        //        DataSet<Tuple2<Integer, DenseVector>> initCentroids = initRandom(points, getK(),
        // getSeed());
        //
        //        IterativeDataSet<Tuple2<Integer, DenseVector>> loopCentroids =
        //                initCentroids.iterate(getMaxIter());
        //
        //        DataSet<Tuple2<Integer, DenseVector>> newCentroids =
        //                points.map(new SelectNearestCentroid())
        //                        .withBroadcastSet(loopCentroids, "centroids")
        //                        .map(new CountAppender())
        //                        .groupBy(0)
        //                        .reduce(new CentroidAccumulator())
        //                        .map(new CentroidAverager());
        //
        //        DataSet<Tuple2<Integer, DenseVector>> finalCentroids =
        //                loopCentroids.closeWith(newCentroids);
        //
        //        return new KMeansModel(finalCentroids);
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static KMeans load(String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
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

    private static class SelectNearestCentroidOperator
            extends AbstractStreamOperator<Tuple2<Integer, DenseVector>>
            implements TwoInputStreamOperator<
                            DenseVector,
                            Tuple2<Integer, DenseVector>,
                            Tuple2<Integer, DenseVector>>,
                    BoundedMultiInput {

        List<DenseVector> points = new ArrayList<>();
        List<Tuple2<Integer, DenseVector>> centroids = new ArrayList<>();
        int numEndedInputs = 0;

        @Override
        public void endInput(int i) throws Exception {
            numEndedInputs++;
            if (numEndedInputs == 2) {
                for (DenseVector point : points) {
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
                    output.collect(new StreamRecord<>(Tuple2.of(closestCentroidId, point)));
                }
            }
        }

        @Override
        public void processElement1(StreamRecord<DenseVector> streamRecord) throws Exception {
            points.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<Tuple2<Integer, DenseVector>> streamRecord)
                throws Exception {
            centroids.add(streamRecord.getValue());
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

    private static class RoundBasedTerminationCriteria
            implements FlatMapFunction<DenseVector, Integer>, IterationListener<Integer> {
        private final int maxRound;

        public RoundBasedTerminationCriteria(int maxRound) {
            this.maxRound = maxRound;
        }

        @Override
        public void flatMap(DenseVector integer, Collector<Integer> collector) throws Exception {}

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<Integer> out) {
            if (epochWatermark < maxRound) {
                out.collect(0);
            }
        }

        @Override
        public void onIterationTerminated(Context context, Collector<Integer> collector) {}
    }

    private static DataStream<Tuple2<Integer, DenseVector>> initRandom(
            DataStream<DenseVector> data, int k, long seed) {
        return data.transform(
                        "initRandom",
                        new TupleTypeInfo<Tuple2<Integer, DenseVector>>(
                                BasicTypeInfo.INT_TYPE_INFO, TypeInformation.of(DenseVector.class)),
                        new MapPartitionFunctionWrapper<>(
                                "initRandom",
                                TypeInformation.of(DenseVector.class),
                                new MapPartitionFunction<
                                        DenseVector, Tuple2<Integer, DenseVector>>() {
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
                                }))
                .setParallelism(1);
    }
}
