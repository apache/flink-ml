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
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.ResultTypeQueryable;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.distance.DistanceMeasure;
import org.apache.flink.ml.common.distance.EuclideanDistanceMeasure;
import org.apache.flink.ml.linalg.VectorWithNorm;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.ml.util.RowUtils;
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
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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

    /**
     * The implementation is based on the nearest-neighbor-chain method proposed in "Modern
     * hierarchical, agglomerative clustering algorithms", by Daniel Mullner.
     */
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

            if (numDataPoints == 0) {
                return;
            }

            DistanceMatrix distanceMatrix = new DistanceMatrix(numDataPoints * 2 - 1);
            VectorWithNorm v1, v2;
            for (int i = 0; i < numDataPoints; i++) {
                v1 = new VectorWithNorm(inputList.get(i).getFieldAs(featuresCol));
                for (int j = i + 1; j < numDataPoints; j++) {
                    v2 = new VectorWithNorm(inputList.get(j).getFieldAs(featuresCol));
                    distanceMatrix.set(i, j, distanceMeasure.distance(v1, v2));
                }
            }

            HashSet<Integer> nodeLabels = new HashSet<>(numDataPoints);
            for (int i = 0; i < numDataPoints; i++) {
                nodeLabels.add(i);
            }

            Tuple2<List<Tuple4<Integer, Integer, Integer, Double>>, int[]> nnChainAndSize =
                    nnChainCore(nodeLabels, distanceMatrix, linkage);

            List<Tuple4<Integer, Integer, Integer, Double>> nnChain = nnChainAndSize.f0;
            nnChain.sort(Comparator.comparingDouble(o -> o.f3));
            reOrderNnChain(nnChain);

            int stoppedIdx = 0;
            if (distanceThreshold != null) {
                for (Tuple4<Integer, Integer, Integer, Double> mergeItem : nnChain) {
                    if (mergeItem.f3 <= distanceThreshold) {
                        stoppedIdx++;
                    }
                }
            } else {
                stoppedIdx = numDataPoints - numCluster;
            }
            List<Tuple4<Integer, Integer, Integer, Double>> earlyStoppedNnChain =
                    nnChain.subList(0, stoppedIdx);

            int[] clusterIds = label(earlyStoppedNnChain, nnChain.size() + 1);

            // Remaps the cluster Ids and output clustering results.
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
                output.collect(RowUtils.append(inputList.get(i), clusterIds[i]));
            }

            // Outputs the merge info.
            if (computeFullTree) {
                stoppedIdx = nnChain.size();
            }
            for (int i = 0; i < stoppedIdx; i++) {
                Tuple4<Integer, Integer, Integer, Double> mergeItem = nnChain.get(i);
                int cid1 = Math.min(mergeItem.f0, mergeItem.f1);
                int cid2 = Math.max(mergeItem.f0, mergeItem.f1);
                context.output(
                        mergeInfoOutputTag,
                        Tuple4.of(
                                cid1,
                                cid2,
                                mergeItem.f3,
                                nnChainAndSize.f1[cid1] + nnChainAndSize.f1[cid2]));
            }
        }

        /** Reorders the nearest-neighbor-chain. */
        private void reOrderNnChain(List<Tuple4<Integer, Integer, Integer, Double>> nnChain) {
            int nextClusterId = nnChain.size() + 1;
            HashMap<Integer, Integer> nodeMapping = new HashMap<>();
            for (Tuple4<Integer, Integer, Integer, Double> t : nnChain) {
                if (nodeMapping.containsKey(t.f0)) {
                    t.f0 = nodeMapping.get(t.f0);
                }
                if (nodeMapping.containsKey(t.f1)) {
                    t.f1 = nodeMapping.get(t.f1);
                }
                nodeMapping.put(t.f2, nextClusterId);
                nextClusterId++;
            }
        }

        /** Converts the cluster Ids for each input data point. */
        private int[] label(
                List<Tuple4<Integer, Integer, Integer, Double>> nnChains, int numDataPoints) {
            UnionFind unionFind = new UnionFind(numDataPoints);
            for (Tuple4<Integer, Integer, Integer, Double> t : nnChains) {
                unionFind.union(unionFind.find(t.f0), unionFind.find(t.f1));
            }
            int[] clusterIds = new int[numDataPoints];
            for (int i = 0; i < clusterIds.length; i++) {
                clusterIds[i] = unionFind.find(i);
            }
            return clusterIds;
        }

        /** The main logic of nearest-neighbor-chain algorithm. */
        private Tuple2<List<Tuple4<Integer, Integer, Integer, Double>>, int[]> nnChainCore(
                HashSet<Integer> nodeLabels, DistanceMatrix distanceMatrix, String linkage) {
            int numDataPoints = nodeLabels.size();
            int nextClusterId = numDataPoints;
            List<Tuple4<Integer, Integer, Integer, Double>> nnChain =
                    new ArrayList<>(numDataPoints);
            List<Integer> chain = new ArrayList<>();
            int[] size = new int[numDataPoints * 2 - 1];
            for (int i = 0; i < numDataPoints; i++) {
                size[i] = 1;
            }

            int a, b;
            while (nodeLabels.size() > 1) {
                if (chain.size() <= 3) {
                    Iterator<Integer> iterator = nodeLabels.iterator();
                    a = iterator.next();
                    chain.clear();
                    chain.add(a);
                    b = iterator.next();
                } else {
                    int chainSize = chain.size();
                    a = chain.get(chainSize - 4);
                    b = chain.get(chainSize - 3);
                    chain.remove(chainSize - 1);
                    chain.remove(chainSize - 2);
                    chain.remove(chainSize - 3);
                }

                while (chain.size() < 3 || chain.get(chain.size() - 3) != a) {
                    double minDistance = Double.MAX_VALUE;
                    int c = -1;
                    for (int x : nodeLabels) {
                        if (x == a) {
                            continue;
                        }
                        double dax = distanceMatrix.get(a, x);
                        if (dax < minDistance) {
                            c = x;
                            minDistance = dax;
                        }
                    }
                    if (minDistance == distanceMatrix.get(a, b) && nodeLabels.contains(b)) {
                        c = b;
                    }
                    b = a;
                    a = c;
                    chain.add(a);
                }

                int mergedNodeLabel = nextClusterId;
                nnChain.add(Tuple4.of(a, b, mergedNodeLabel, distanceMatrix.get(a, b)));
                nodeLabels.remove(a);
                nodeLabels.remove(b);
                nextClusterId++;
                size[mergedNodeLabel] = size[a] + size[b];

                for (int x : nodeLabels) {
                    double d =
                            computeClusterDistances(
                                    distanceMatrix.get(a, x),
                                    distanceMatrix.get(b, x),
                                    distanceMatrix.get(a, b),
                                    size[a],
                                    size[b],
                                    size[x],
                                    linkage);
                    distanceMatrix.set(x, mergedNodeLabel, d);
                }

                nodeLabels.add(mergedNodeLabel);
            }

            return Tuple2.of(nnChain, size);
        }

        /** Utility class for finding labels for input data points. */
        private static class UnionFind {
            private final int[] parent;
            private int nextLabel;

            public UnionFind(int numDataPoints) {
                parent = new int[2 * numDataPoints - 1];
                Arrays.fill(parent, -1);
                nextLabel = numDataPoints;
            }

            public void union(int m, int n) {
                parent[m] = nextLabel;
                parent[n] = nextLabel;
                nextLabel++;
            }

            public int find(int n) {
                int p = n;
                while (parent[n] != -1) {
                    n = parent[n];
                }
                while (parent[p] != n && parent[p] != -1) {
                    p = parent[p];
                    parent[p] = n;
                }
                return n;
            }
        }

        /** Utility class for storing distances between every two clusters. */
        private static class DistanceMatrix {
            /** The storage of distances between each two clusters. */
            private final double[] distances;
            /** Number of clusters. */
            private final int n;

            public DistanceMatrix(int n) {
                distances = new double[n * (n - 1) / 2];
                this.n = n;
            }

            public void set(int i, int j, double value) {
                int smallIdx = Math.min(i, j);
                int bigIdx = Math.max(i, j);
                int offset = (n * 2 - 1 - smallIdx) * smallIdx / 2 + (bigIdx - smallIdx - 1);
                distances[offset] = value;
            }

            public double get(int i, int j) {
                int smallIdx = Math.min(i, j);
                int bigIdx = Math.max(i, j);
                int offset = (n * 2 - 1 - smallIdx) * smallIdx / 2 + (bigIdx - smallIdx - 1);
                return distances[offset];
            }
        }

        /**
         * Computes the distance between cluster k and the new cluster merged by cluster i and j.
         *
         * @param dik distance between cluster i and k.
         * @param djk distance between cluster j and k.
         * @param dij distance between cluster i and j.
         * @param si size of cluster i.
         * @param sj size of cluster j.
         * @param sk size of cluster k.
         * @param linkage the linkage method.
         * @return distance between cluster k and the newly merged cluster.
         */
        private double computeClusterDistances(
                double dik, double djk, double dij, int si, int sj, int sk, String linkage) {
            switch (linkage) {
                case LINKAGE_SINGLE:
                    return Math.min(dik, djk);
                case LINKAGE_COMPLETE:
                    return Math.max(dik, djk);
                case LINKAGE_AVERAGE:
                    return (si * dik + sj * djk) / (si + sj);
                case LINKAGE_WARD:
                    return Math.sqrt(
                            ((si + sk) * dik * dik + (sj + sk) * djk * djk - sk * dij * dij)
                                    / (si + sj + sk));
                default:
                    throw new UnsupportedOperationException(
                            "Unsupported " + LINKAGE + " type: " + linkage + ".");
            }
        }

        @Override
        public TypeInformation<Row> getProducedType() {
            return outputTypeInfo;
        }
    }
}
