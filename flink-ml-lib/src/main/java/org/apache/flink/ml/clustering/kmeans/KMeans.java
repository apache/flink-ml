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
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicArrayTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.distance.DistanceMeasure;
import org.apache.flink.ml.common.iteration.ForwardInputsOfLastRound;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.VectorWithNorm;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorWithNormSerializer;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * An Estimator which implements the k-means clustering algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/K-means_clustering.
 */
public class KMeans implements Estimator<KMeans, KMeansModel>, KMeansParams<KMeans> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public KMeans() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public KMeansModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<DenseIntDoubleVector> points =
                tEnv.toDataStream(inputs[0])
                        .map(row -> ((IntDoubleVector) row.getField(getFeaturesCol())).toDense());

        DataStream<DenseIntDoubleVector[]> initCentroids =
                selectRandomCentroids(points, getK(), getSeed());

        IterationConfig config =
                IterationConfig.newBuilder()
                        .setOperatorLifeCycle(IterationConfig.OperatorLifeCycle.ALL_ROUND)
                        .build();

        IterationBody body =
                new KMeansIterationBody(
                        getMaxIter(), DistanceMeasure.getInstance(getDistanceMeasure()));

        DataStream<KMeansModelData> finalModelData =
                Iterations.iterateBoundedStreamsUntilTermination(
                                DataStreamList.of(initCentroids),
                                ReplayableDataStreamList.notReplay(points),
                                config,
                                body)
                        .get(0);

        Table finalModelDataTable = tEnv.fromDataStream(finalModelData);
        KMeansModel model = new KMeansModel().setModelData(finalModelDataTable);
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static KMeans load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class KMeansIterationBody implements IterationBody {
        private final int maxIterationNum;
        private final DistanceMeasure distanceMeasure;

        public KMeansIterationBody(int maxIterationNum, DistanceMeasure distanceMeasure) {
            this.maxIterationNum = maxIterationNum;
            this.distanceMeasure = distanceMeasure;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<DenseIntDoubleVector[]> centroids = variableStreams.get(0);
            DataStream<DenseIntDoubleVector> points = dataStreams.get(0);

            DataStream<Integer> terminationCriteria =
                    centroids.flatMap(new TerminateOnMaxIter(maxIterationNum));

            DataStream<Tuple2<Integer[], DenseIntDoubleVector[]>> centroidIdAndPoints =
                    points.connect(centroids.broadcast())
                            .transform(
                                    "CentroidsUpdateAccumulator",
                                    new TupleTypeInfo<>(
                                            BasicArrayTypeInfo.INT_ARRAY_TYPE_INFO,
                                            ObjectArrayTypeInfo.getInfoFor(
                                                    DenseIntDoubleVectorTypeInfo.INSTANCE)),
                                    new CentroidsUpdateAccumulator(distanceMeasure));

            DataStreamUtils.setManagedMemoryWeight(centroidIdAndPoints, 100);

            int parallelism = centroidIdAndPoints.getParallelism();
            DataStream<KMeansModelData> newModelData =
                    centroidIdAndPoints
                            .countWindowAll(parallelism)
                            .reduce(new CentroidsUpdateReducer())
                            .map(new ModelDataGenerator());

            DataStream<DenseIntDoubleVector[]> newCentroids =
                    newModelData.map(x -> x.centroids).setParallelism(1);

            DataStream<KMeansModelData> finalModelData =
                    newModelData.flatMap(new ForwardInputsOfLastRound<>());

            return new IterationBodyResult(
                    DataStreamList.of(newCentroids),
                    DataStreamList.of(finalModelData),
                    terminationCriteria);
        }
    }

    private static class CentroidsUpdateReducer
            implements ReduceFunction<Tuple2<Integer[], DenseIntDoubleVector[]>> {
        @Override
        public Tuple2<Integer[], DenseIntDoubleVector[]> reduce(
                Tuple2<Integer[], DenseIntDoubleVector[]> tuple2,
                Tuple2<Integer[], DenseIntDoubleVector[]> t1)
                throws Exception {
            for (int i = 0; i < tuple2.f0.length; i++) {
                tuple2.f0[i] += t1.f0[i];
                BLAS.axpy(1.0, t1.f1[i], tuple2.f1[i]);
            }

            return tuple2;
        }
    }

    private static class ModelDataGenerator
            implements MapFunction<Tuple2<Integer[], DenseIntDoubleVector[]>, KMeansModelData> {
        @Override
        public KMeansModelData map(Tuple2<Integer[], DenseIntDoubleVector[]> tuple2)
                throws Exception {
            double[] weights = new double[tuple2.f0.length];
            for (int i = 0; i < tuple2.f0.length; i++) {
                BLAS.scal(1.0 / tuple2.f0[i], tuple2.f1[i]);
                weights[i] = tuple2.f0[i];
            }

            return new KMeansModelData(tuple2.f1, new DenseIntDoubleVector(weights));
        }
    }

    private static class CentroidsUpdateAccumulator
            extends AbstractStreamOperator<Tuple2<Integer[], DenseIntDoubleVector[]>>
            implements TwoInputStreamOperator<
                            DenseIntDoubleVector,
                            DenseIntDoubleVector[],
                            Tuple2<Integer[], DenseIntDoubleVector[]>>,
                    IterationListener<Tuple2<Integer[], DenseIntDoubleVector[]>> {

        private final DistanceMeasure distanceMeasure;

        private ListState<DenseIntDoubleVector[]> centroids;

        private ListStateWithCache<VectorWithNorm> points;

        public CentroidsUpdateAccumulator(DistanceMeasure distanceMeasure) {
            super();
            this.distanceMeasure = distanceMeasure;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            TypeInformation<DenseIntDoubleVector[]> type =
                    ObjectArrayTypeInfo.getInfoFor(DenseIntDoubleVectorTypeInfo.INSTANCE);

            centroids =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("centroids", type));

            points =
                    new ListStateWithCache<>(
                            new VectorWithNormSerializer(),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            points.snapshotState(context);
        }

        @Override
        public void processElement1(StreamRecord<DenseIntDoubleVector> streamRecord)
                throws Exception {
            points.add(new VectorWithNorm(streamRecord.getValue()));
        }

        @Override
        public void processElement2(StreamRecord<DenseIntDoubleVector[]> streamRecord)
                throws Exception {
            Preconditions.checkState(!centroids.get().iterator().hasNext());
            centroids.add(streamRecord.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark,
                Context context,
                Collector<Tuple2<Integer[], DenseIntDoubleVector[]>> out)
                throws Exception {
            DenseIntDoubleVector[] centroidValues =
                    Objects.requireNonNull(
                            OperatorStateUtils.getUniqueElement(centroids, "centroids")
                                    .orElse(null));

            VectorWithNorm[] centroidsWithNorm = new VectorWithNorm[centroidValues.length];
            for (int i = 0; i < centroidsWithNorm.length; i++) {
                centroidsWithNorm[i] = new VectorWithNorm(centroidValues[i]);
            }

            DenseIntDoubleVector[] newCentroids = new DenseIntDoubleVector[centroidValues.length];
            Integer[] counts = new Integer[centroidValues.length];
            Arrays.fill(counts, 0);
            for (int i = 0; i < centroidValues.length; i++) {
                newCentroids[i] = new DenseIntDoubleVector(centroidValues[i].size());
            }

            for (VectorWithNorm point : points.get()) {
                int closestCentroidId = distanceMeasure.findClosest(centroidsWithNorm, point);
                BLAS.axpy(1.0, point.vector, newCentroids[closestCentroidId]);
                counts[closestCentroidId]++;
            }

            output.collect(new StreamRecord<>(Tuple2.of(counts, newCentroids)));

            centroids.clear();
        }

        @Override
        public void onIterationTerminated(
                Context context, Collector<Tuple2<Integer[], DenseIntDoubleVector[]>> collector) {
            centroids.clear();
            points.clear();
        }
    }

    public static DataStream<DenseIntDoubleVector[]> selectRandomCentroids(
            DataStream<DenseIntDoubleVector> data, int k, long seed) {
        DataStream<DenseIntDoubleVector[]> resultStream =
                DataStreamUtils.mapPartition(
                        DataStreamUtils.sample(data, k, seed),
                        new MapPartitionFunction<DenseIntDoubleVector, DenseIntDoubleVector[]>() {
                            @Override
                            public void mapPartition(
                                    Iterable<DenseIntDoubleVector> iterable,
                                    Collector<DenseIntDoubleVector[]> collector) {
                                List<DenseIntDoubleVector> list = new ArrayList<>();
                                iterable.iterator().forEachRemaining(list::add);
                                collector.collect(list.toArray(new DenseIntDoubleVector[0]));
                            }
                        });
        resultStream.getTransformation().setParallelism(1);
        return resultStream;
    }
}
