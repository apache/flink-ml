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

package org.apache.flink.ml.feature.kbinsdiscretizer;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.feature.minmaxscaler.MinMaxScaler.MinMaxReduceFunctionOperator;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An Estimator which implements discretization (also known as quantization or binning) to transform
 * continuous features into discrete ones. The output values are in [0, numBins).
 *
 * <p>KBinsDiscretizer implements three different binning strategies, and it can be set by {@link
 * KBinsDiscretizerParams#STRATEGY}. If the strategy is set as {@link KBinsDiscretizerParams#KMEANS}
 * or {@link KBinsDiscretizerParams#QUANTILE}, users should further set {@link
 * KBinsDiscretizerParams#SUB_SAMPLES} for better performance.
 *
 * <p>There are several corner cases for different inputs as listed below:
 *
 * <ul>
 *   <li>When the input values of one column are all the same, then they should be mapped to the
 *       same bin (i.e., the zero-th bin). Thus the corresponding bin edges are
 *       `{Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY}`.
 *   <li>When the number of distinct values of one column is less than the specified number of bins
 *       and the {@link KBinsDiscretizerParams#STRATEGY} is set as {@link
 *       KBinsDiscretizerParams#KMEANS}, we switch to {@link KBinsDiscretizerParams#UNIFORM}.
 *   <li>When the width of one output bin is zero, i.e., the left edge equals to the right edge of
 *       the bin, we replace the right edge as the average value of its two neighbors. One exception
 *       is that the last two edges are the same --- in this case, the left edge is updated as the
 *       average of its two neighbors. For example, the bin edges {0, 1, 1, 2, 2} are transformed
 *       into {0, 1, 1.5, 1.75, 2}.
 * </ul>
 */
public class KBinsDiscretizer
        implements Estimator<KBinsDiscretizer, KBinsDiscretizerModel>,
                KBinsDiscretizerParams<KBinsDiscretizer> {
    private static final Logger LOG = LoggerFactory.getLogger(KBinsDiscretizer.class);
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public KBinsDiscretizer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public KBinsDiscretizerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        String inputCol = getInputCol();
        String strategy = getStrategy();
        int numBins = getNumBins();

        DataStream<DenseIntDoubleVector> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, DenseIntDoubleVector>)
                                        value ->
                                                ((IntDoubleVector) value.getField(inputCol))
                                                        .toDense());

        DataStream<DenseIntDoubleVector> preprocessedData;
        if (strategy.equals(UNIFORM)) {
            preprocessedData =
                    inputData
                            .transform(
                                    "reduceInEachPartition",
                                    inputData.getType(),
                                    new MinMaxReduceFunctionOperator())
                            .transform(
                                    "reduceInFinalPartition",
                                    inputData.getType(),
                                    new MinMaxReduceFunctionOperator())
                            .setParallelism(1);
        } else {
            preprocessedData =
                    DataStreamUtils.sample(
                            inputData, getSubSamples(), getClass().getName().hashCode());
        }

        DataStream<KBinsDiscretizerModelData> modelData =
                DataStreamUtils.mapPartition(
                        preprocessedData,
                        new MapPartitionFunction<
                                DenseIntDoubleVector, KBinsDiscretizerModelData>() {
                            @Override
                            public void mapPartition(
                                    Iterable<DenseIntDoubleVector> iterable,
                                    Collector<KBinsDiscretizerModelData> collector) {
                                List<DenseIntDoubleVector> list = new ArrayList<>();
                                iterable.iterator().forEachRemaining(list::add);

                                if (list.size() == 0) {
                                    throw new RuntimeException("The training set is empty.");
                                }

                                double[][] binEdges;
                                switch (strategy) {
                                    case UNIFORM:
                                        binEdges = findBinEdgesWithUniformStrategy(list, numBins);
                                        break;
                                    case QUANTILE:
                                        binEdges = findBinEdgesWithQuantileStrategy(list, numBins);
                                        break;
                                    case KMEANS:
                                        binEdges = findBinEdgesWithKMeansStrategy(list, numBins);
                                        break;
                                    default:
                                        throw new UnsupportedOperationException(
                                                "Unsupported "
                                                        + STRATEGY
                                                        + " type: "
                                                        + strategy
                                                        + ".");
                                }

                                collector.collect(new KBinsDiscretizerModelData(binEdges));
                            }
                        });
        modelData.getTransformation().setParallelism(1);

        KBinsDiscretizerModel model =
                new KBinsDiscretizerModel().setModelData(tEnv.fromDataStream(modelData));
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

    public static KBinsDiscretizer load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    private static double[][] findBinEdgesWithUniformStrategy(
            List<DenseIntDoubleVector> input, int numBins) {
        DenseIntDoubleVector minVector = input.get(0);
        DenseIntDoubleVector maxVector = input.get(1);
        int numColumns = minVector.size();
        double[][] binEdges = new double[numColumns][];

        for (int columnId = 0; columnId < numColumns; columnId++) {
            double min = minVector.get(columnId);
            double max = maxVector.get(columnId);
            if (min == max) {
                LOG.warn("Feature " + columnId + " is constant and the output will all be zero.");
                binEdges[columnId] =
                        new double[] {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY};
            } else {
                double width = (max - min) / numBins;
                binEdges[columnId] = new double[numBins + 1];
                binEdges[columnId][0] = min;
                for (int edgeId = 1; edgeId < numBins + 1; edgeId++) {
                    binEdges[columnId][edgeId] = binEdges[columnId][edgeId - 1] + width;
                }
            }
        }
        return binEdges;
    }

    private static double[][] findBinEdgesWithQuantileStrategy(
            List<DenseIntDoubleVector> input, int numBins) {
        int numColumns = input.get(0).size();
        int numData = input.size();
        double[][] binEdges = new double[numColumns][];
        double[] features = new double[numData];

        for (int columnId = 0; columnId < numColumns; columnId++) {
            for (int i = 0; i < numData; i++) {
                features[i] = input.get(i).get(columnId);
            }
            Arrays.sort(features);

            if (features[0] == features[numData - 1]) {
                LOG.warn("Feature " + columnId + " is constant and the output will all be zero.");
                binEdges[columnId] =
                        new double[] {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY};
            } else {
                double[] tempBinEdges;
                if (features.length > numBins) {
                    double width = 1.0 * features.length / numBins;
                    tempBinEdges = new double[numBins + 1];

                    for (int binEdgeId = 0; binEdgeId < numBins; binEdgeId++) {
                        tempBinEdges[binEdgeId] = features[(int) (binEdgeId * width)];
                    }
                    tempBinEdges[numBins] = features[numData - 1];
                } else {
                    tempBinEdges = features;
                }

                // Bins with zero width should be converted to a non-empty bin.
                Map<Double, Integer> edgesAndCnt = new HashMap<>(numBins);
                for (double edge : tempBinEdges) {
                    edgesAndCnt.put(edge, edgesAndCnt.getOrDefault(edge, 0) + 1);
                }
                List<Double> edges = new ArrayList<>();
                for (Map.Entry<Double, Integer> edgeAndCnt : edgesAndCnt.entrySet()) {
                    double edge = edgeAndCnt.getKey();
                    int cnt = edgeAndCnt.getValue();
                    edges.add(edge);
                    if (cnt > 1) {
                        edges.add(edge);
                    }
                }
                tempBinEdges = edges.stream().mapToDouble(Double::doubleValue).toArray();
                Arrays.sort(tempBinEdges);
                int i = 1;
                // If there are two consecutive bin edges with the same value, we update the right
                // edge as the average of its two neighbors.
                for (; i < tempBinEdges.length - 1; i++) {
                    if (tempBinEdges[i] == tempBinEdges[i - 1]) {
                        tempBinEdges[i] = (tempBinEdges[i + 1] + tempBinEdges[i - 1]) / 2;
                    }
                }
                // If the last two bin edges are the same, we update the left bin edge as the
                // average of its two neighbors.
                if (tempBinEdges[i] == tempBinEdges[i - 1]) {
                    tempBinEdges[i - 1] = (tempBinEdges[i] + tempBinEdges[i - 2]) / 2;
                }
                binEdges[columnId] = tempBinEdges;
            }
        }
        return binEdges;
    }

    private static double[][] findBinEdgesWithKMeansStrategy(
            List<DenseIntDoubleVector> input, int numBins) {
        int numColumns = input.get(0).size();
        int numData = input.size();
        double[][] binEdges = new double[numColumns][numBins + 1];
        double[] features = new double[numData];

        double[] kMeansCentroids = new double[numBins];
        double[] sumByCluster = new double[numBins];

        for (int columnId = 0; columnId < numColumns; columnId++) {
            for (int i = 0; i < numData; i++) {
                features[i] = input.get(i).get(columnId);
            }
            Arrays.sort(features);

            if (features[0] == features[numData - 1]) {
                LOG.warn("Feature " + columnId + " is constant and the output will all be zero.");
                binEdges[columnId] =
                        new double[] {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY};
            } else {
                // Checks whether there are more than {numBins} distinct feature values in each
                // column.
                // If the number of distinct values is less than {numBins + 1}, then we do not need
                // to conduct KMeans. Instead, we switch to using {@link
                // KBinsDiscretizerParams#UNIFORM} for binning.
                Set<Double> distinctFeatureValues = new HashSet<>(numBins + 1);
                for (double feature : features) {
                    distinctFeatureValues.add(feature);
                    if (distinctFeatureValues.size() >= numBins + 1) {
                        break;
                    }
                }
                if (distinctFeatureValues.size() <= numBins) {
                    double min = features[0];
                    double max = features[features.length - 1];
                    double width = (max - min) / numBins;
                    binEdges[columnId] = new double[numBins + 1];
                    binEdges[columnId][0] = min;
                    for (int edgeId = 1; edgeId < numBins + 1; edgeId++) {
                        binEdges[columnId][edgeId] = binEdges[columnId][edgeId - 1] + width;
                    }
                    continue;
                } else {
                    // Conducts KMeans here.
                    double width = 1.0 * features.length / numBins;
                    for (int clusterId = 0; clusterId < numBins; clusterId++) {
                        kMeansCentroids[clusterId] = features[(int) (clusterId * width)];
                    }

                    // Default values for KMeans.
                    final double tolerance = 1e-4;
                    final int maxIterations = 300;

                    double oldLoss = Double.MAX_VALUE;
                    double relativeLoss = Double.MAX_VALUE;
                    int iter = 0;
                    int[] countByCluster = new int[numBins];
                    while (iter < maxIterations && relativeLoss > tolerance) {
                        double loss = 0;
                        for (double featureValue : features) {
                            double minDistance = Math.abs(kMeansCentroids[0] - featureValue);
                            int clusterId = 0;
                            for (int i = 1; i < kMeansCentroids.length; i++) {
                                double distance = Math.abs(kMeansCentroids[i] - featureValue);
                                if (distance < minDistance) {
                                    minDistance = distance;
                                    clusterId = i;
                                }
                            }
                            countByCluster[clusterId]++;
                            sumByCluster[clusterId] += featureValue;
                            loss += minDistance;
                        }

                        // Updates cluster.
                        for (int clusterId = 0; clusterId < kMeansCentroids.length; clusterId++) {
                            kMeansCentroids[clusterId] =
                                    sumByCluster[clusterId] / countByCluster[clusterId];
                        }
                        loss /= features.length;
                        relativeLoss = Math.abs(loss - oldLoss);
                        oldLoss = loss;
                        iter++;
                        Arrays.fill(sumByCluster, 0);
                        Arrays.fill(countByCluster, 0);
                    }

                    Arrays.sort(kMeansCentroids);
                    binEdges[columnId] = new double[numBins + 1];
                    binEdges[columnId][0] = features[0];
                    binEdges[columnId][numBins] = features[features.length - 1];
                    for (int binEdgeId = 1; binEdgeId < numBins; binEdgeId++) {
                        binEdges[columnId][binEdgeId] =
                                (kMeansCentroids[binEdgeId - 1] + kMeansCentroids[binEdgeId]) / 2;
                    }
                }
            }
        }
        return binEdges;
    }
}
