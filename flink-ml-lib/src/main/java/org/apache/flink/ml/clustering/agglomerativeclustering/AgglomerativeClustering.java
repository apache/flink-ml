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

package org.apache.flink.ml.clustering.agglomerativeclustering;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.ResultTypeQueryable;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.distance.DistanceMeasure;
import org.apache.flink.ml.common.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.VectorWithNorm;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.functions.windowing.ProcessAllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.Window;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An AlgoOperator that performs a hierarchical clustering using a bottom-up approach. Each
 * observation starts in its own cluster and the clusters are merged together one by one. Users can
 * choose different strategies to merge two clusters by setting {@link
 * AgglomerativeClusteringParams#LINKAGE} and different distance measures by setting {@link
 * AgglomerativeClusteringParams#DISTANCE_MEASURE}.
 *
 * <p>The output contains two tables. The first one assigns one cluster Id for each data point. The
 * second one contains the information of merging two clusters at each step. The data format of the
 * merging information is (clusterId1, clusterId2, distance, sizeOfMergedCluster).
 *
 * <p>This AlgoOperator splits input stream into mini-batches of elements according to the windowing
 * strategy specified by the {@link org.apache.flink.ml.common.param.HasWindows} parameter, and
 * performs the hierarchical clustering on each mini-batch independently. The clustering result of
 * each element depends only on the elements in the same mini-batch.
 *
 * <p>See https://en.wikipedia.org/wiki/Hierarchical_clustering.
 */
public class AgglomerativeClustering
        implements AlgoOperator<AgglomerativeClustering>,
                AgglomerativeClusteringParams<AgglomerativeClustering> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public AgglomerativeClustering() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Integer numCluster = getNumClusters();
        Double distanceThreshold = getDistanceThreshold();
        Preconditions.checkArgument(
                (numCluster == null && distanceThreshold != null)
                        || (numCluster != null && distanceThreshold == null),
                "One of param numCluster and distanceThreshold should be null.");

        if (getLinkage().equals(LINKAGE_WARD)) {
            String distanceMeasure = getDistanceMeasure();
            Preconditions.checkArgument(
                    distanceMeasure.equals(EuclideanDistanceMeasure.NAME),
                    distanceMeasure
                            + " was provided as distance measure while linkage was ward. Ward only works with euclidean.");
        }

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> dataStream = tEnv.toDataStream(inputs[0]);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), Types.INT),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol()));

        OutputTag<Tuple4<Integer, Integer, Double, Integer>> mergeInfoOutputTag =
                new OutputTag<Tuple4<Integer, Integer, Double, Integer>>("MERGE_INFO") {};

        SingleOutputStreamOperator<Row> output =
                DataStreamUtils.windowAllAndProcess(
                        dataStream,
                        getWindows(),
                        new LocalAgglomerativeClusteringFunction<>(
                                getFeaturesCol(),
                                getLinkage(),
                                getDistanceMeasure(),
                                getNumClusters(),
                                getDistanceThreshold(),
                                getComputeFullTree(),
                                mergeInfoOutputTag,
                                outputTypeInfo));

        Schema schema =
                Schema.newBuilder()
                        .fromResolvedSchema(inputs[0].getResolvedSchema())
                        .column(getPredictionCol(), DataTypes.INT())
                        .build();

        Table outputTable = tEnv.fromDataStream(output, schema);

        DataStream<Tuple4<Integer, Integer, Double, Integer>> mergeInfo =
                output.getSideOutput(mergeInfoOutputTag);
        mergeInfo.getTransformation().setParallelism(1);
        Table mergeInfoTable =
                tEnv.fromDataStream(mergeInfo)
                        .as("clusterId1", "clusterId2", "distance", "sizeOfMergedCluster");

        return new Table[] {outputTable, mergeInfoTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static AgglomerativeClustering load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class LocalAgglomerativeClusteringFunction<W extends Window>
            extends ProcessAllWindowFunction<Row, Row, W> implements ResultTypeQueryable<Row> {
        private final String featuresCol;
        private final String linkage;
        private final DistanceMeasure distanceMeasure;
        private final Integer numCluster;
        private final Double distanceThreshold;
        private final boolean computeFullTree;
        private final OutputTag<Tuple4<Integer, Integer, Double, Integer>> mergeInfoOutputTag;
        private final RowTypeInfo outputTypeInfo;

        /** Cluster id of each data point in inputList. */
        private int[] clusterIds;
        /** Precomputes the norm of each vector for performance. */
        private VectorWithNorm[] vectorWithNorms;
        /** Next cluster Id to be assigned. */
        private int nextClusterId = 0;

        public LocalAgglomerativeClusteringFunction(
                String featuresCol,
                String linkage,
                String distanceMeasureName,
                Integer numCluster,
                Double distanceThreshold,
                boolean computeFullTree,
                OutputTag<Tuple4<Integer, Integer, Double, Integer>> mergeInfoOutputTag,
                RowTypeInfo outputTypeInfo) {
            this.featuresCol = featuresCol;
            this.linkage = linkage;
            this.numCluster = numCluster;
            this.distanceThreshold = distanceThreshold;
            this.computeFullTree = computeFullTree;
            this.mergeInfoOutputTag = mergeInfoOutputTag;

            distanceMeasure = DistanceMeasure.getInstance(distanceMeasureName);
            this.outputTypeInfo = outputTypeInfo;
        }

        @Override
        public void process(
                ProcessAllWindowFunction<Row, Row, W>.Context context,
                Iterable<Row> values,
                Collector<Row> output) {
            List<Row> inputList = IteratorUtils.toList(values.iterator());
            int numDataPoints = inputList.size();

            // Assigns initial cluster Ids.
            clusterIds = new int[numDataPoints];
            for (int i = 0; i < numDataPoints; i++) {
                clusterIds[i] = getNextClusterId();
            }

            List<Cluster> activeClusters = new ArrayList<>();
            for (int i = 0; i < numDataPoints; i++) {
                List<Integer> dataPointIds = new ArrayList<>();
                dataPointIds.add(i);
                activeClusters.add(new Cluster(i, dataPointIds));
            }

            // Precomputes vector norms for faster computation.
            vectorWithNorms = new VectorWithNorm[inputList.size()];
            for (int i = 0; i < numDataPoints; i++) {
                vectorWithNorms[i] =
                        new VectorWithNorm((Vector) inputList.get(i).getField(featuresCol));
            }

            // Clustering process.
            doClustering(activeClusters, context);

            // Remaps the cluster Ids and output results.
            HashMap<Integer, Integer> remappedClusterIds = new HashMap<>();
            int cnt = 0;
            for (int i = 0; i < clusterIds.length; i++) {
                int clusterId = clusterIds[i];
                if (remappedClusterIds.containsKey(clusterId)) {
                    clusterIds[i] = remappedClusterIds.get(clusterId);
                } else {
                    clusterIds[i] = cnt;
                    remappedClusterIds.put(clusterId, cnt++);
                }
            }

            for (int i = 0; i < numDataPoints; i++) {
                output.collect(Row.join(inputList.get(i), Row.of(clusterIds[i])));
            }
        }

        private int getNextClusterId() {
            return nextClusterId++;
        }

        private void doClustering(
                List<Cluster> activeClusters,
                ProcessAllWindowFunction<Row, Row, ?>.Context context) {
            int clusterOffset1 = -1, clusterOffset2 = -1;
            boolean clusteringRunning =
                    (numCluster != null && activeClusters.size() > numCluster)
                            || (distanceThreshold != null);

            while (clusteringRunning || (computeFullTree && activeClusters.size() > 1)) {
                // Computes the distance between two clusters.
                double minDistance = Double.MAX_VALUE;
                for (int i = 0; i < activeClusters.size(); i++) {
                    for (int j = i + 1; j < activeClusters.size(); j++) {
                        double distance =
                                computeDistanceBetweenClusters(
                                        activeClusters.get(i), activeClusters.get(j));
                        if (distance < minDistance) {
                            minDistance = distance;
                            clusterOffset1 = i;
                            clusterOffset2 = j;
                        }
                    }
                }

                // Outputs the merge info.
                Cluster cluster1 = activeClusters.get(clusterOffset1);
                Cluster cluster2 = activeClusters.get(clusterOffset2);
                int clusterId1 = cluster1.clusterId;
                int clusterId2 = cluster2.clusterId;
                context.output(
                        mergeInfoOutputTag,
                        Tuple4.of(
                                Math.min(clusterId1, clusterId2),
                                Math.max(clusterId1, clusterId2),
                                minDistance,
                                cluster1.dataPointIds.size() + cluster2.dataPointIds.size()));

                // Merges these two clusters.
                Cluster mergedCluster =
                        new Cluster(
                                getNextClusterId(), cluster1.dataPointIds, cluster2.dataPointIds);
                activeClusters.set(clusterOffset1, mergedCluster);
                activeClusters.remove(clusterOffset2);

                // Updates cluster Ids for each data point if clustering is still running.
                if (clusteringRunning) {
                    int mergedClusterId = mergedCluster.clusterId;
                    for (int dataPointId : mergedCluster.dataPointIds) {
                        clusterIds[dataPointId] = mergedClusterId;
                    }
                }

                clusteringRunning =
                        (numCluster != null && activeClusters.size() > numCluster)
                                || (distanceThreshold != null && distanceThreshold > minDistance);
            }
        }

        private double computeDistanceBetweenClusters(Cluster cluster1, Cluster cluster2) {
            double distance;
            int size1 = cluster1.dataPointIds.size();
            int size2 = cluster2.dataPointIds.size();

            switch (linkage) {
                case LINKAGE_AVERAGE:
                    distance = 0;
                    for (int i = 0; i < size1; i++) {
                        for (int j = 0; j < size2; j++) {
                            VectorWithNorm vectorWithNorm1 =
                                    vectorWithNorms[cluster1.dataPointIds.get(i)];
                            VectorWithNorm vectorWithNorm2 =
                                    vectorWithNorms[cluster2.dataPointIds.get(j)];
                            distance += distanceMeasure.distance(vectorWithNorm1, vectorWithNorm2);
                        }
                    }
                    distance /= size1 * size2;
                    break;
                case LINKAGE_COMPLETE:
                    distance = Double.MIN_VALUE;
                    for (int i = 0; i < size1; i++) {
                        for (int j = 0; j < size2; j++) {
                            VectorWithNorm vectorWithNorm1 =
                                    vectorWithNorms[cluster1.dataPointIds.get(i)];
                            VectorWithNorm vectorWithNorm2 =
                                    vectorWithNorms[cluster2.dataPointIds.get(j)];
                            distance =
                                    Math.max(
                                            distance,
                                            distanceMeasure.distance(
                                                    vectorWithNorm1, vectorWithNorm2));
                        }
                    }
                    break;
                case LINKAGE_SINGLE:
                    distance = Double.MAX_VALUE;
                    for (int i = 0; i < size1; i++) {
                        for (int j = 0; j < size2; j++) {
                            VectorWithNorm vectorWithNorm1 =
                                    vectorWithNorms[cluster1.dataPointIds.get(i)];
                            VectorWithNorm vectorWithNorm2 =
                                    vectorWithNorms[cluster2.dataPointIds.get(j)];
                            distance =
                                    Math.min(
                                            distance,
                                            distanceMeasure.distance(
                                                    vectorWithNorm1, vectorWithNorm2));
                        }
                    }
                    break;
                case LINKAGE_WARD:
                    int vecSize = vectorWithNorms[0].vector.size();
                    DenseVector mean1 = Vectors.dense(new double[vecSize]);
                    DenseVector mean2 = Vectors.dense(new double[vecSize]);

                    for (int i = 0; i < size1; i++) {
                        BLAS.axpy(1.0, vectorWithNorms[cluster1.dataPointIds.get(i)].vector, mean1);
                    }
                    for (int i = 0; i < size2; i++) {
                        BLAS.axpy(1.0, vectorWithNorms[cluster2.dataPointIds.get(i)].vector, mean2);
                    }

                    DenseVector meanMerged = mean1.clone();
                    BLAS.axpy(1.0, mean2, meanMerged);
                    BLAS.scal(1.0 / size1, mean1);
                    BLAS.scal(1.0 / size2, mean2);
                    BLAS.scal(1.0 / (size1 + size2), meanMerged);
                    double essInc =
                            size1 * BLAS.dot(mean1, mean1)
                                    + size2 * BLAS.dot(mean2, mean2)
                                    - (size1 + size2) * BLAS.dot(meanMerged, meanMerged);

                    distance = Math.sqrt(2 * essInc);
                    break;
                default:
                    throw new UnsupportedOperationException(
                            "Unsupported " + LINKAGE + " type: " + linkage + ".");
            }
            return distance;
        }

        @Override
        public TypeInformation<Row> getProducedType() {
            return outputTypeInfo;
        }

        /** A cluster with cluster Id specified and data points that belong to this cluster. */
        private static class Cluster {
            private final int clusterId;
            private final List<Integer> dataPointIds;

            public Cluster(int clusterId, List<Integer> dataPointIds) {
                this.clusterId = clusterId;
                this.dataPointIds = dataPointIds;
            }

            public Cluster(
                    int clusterId, List<Integer> dataPointIds, List<Integer> otherDataPointIds) {
                this.clusterId = clusterId;
                this.dataPointIds = dataPointIds;
                this.dataPointIds.addAll(otherDataPointIds);
            }
        }
    }
}
