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

import org.apache.flink.ml.common.param.HasDistanceMeasure;
import org.apache.flink.ml.common.param.HasFeaturesCol;
import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params of {@link AgglomerativeClustering}.
 *
 * @param <T> The class type of this instance.
 */
public interface AgglomerativeClusteringParams<T>
        extends HasDistanceMeasure<T>, HasFeaturesCol<T>, HasPredictionCol<T> {
    Param<Integer> NUM_CLUSTERS =
            new IntParam("numClusters", "The max number of clusters to create.", 2);

    Param<Double> DISTANCE_THRESHOLD =
            new DoubleParam(
                    "distanceThreshold",
                    "Threshold to decide whether two clusters should be merged.",
                    null);

    String LINKAGE_WARD = "ward";
    String LINKAGE_COMPLETE = "complete";
    String LINKAGE_SINGLE = "single";
    String LINKAGE_AVERAGE = "average";
    /**
     * Supported options to compute the distance between two clusters. The algorithm will merge the
     * pairs of cluster that minimize this criterion.
     *
     * <ul>
     *   <li>ward: the variance between the two clusters.
     *   <li>complete: the maximum distance between all observations of the two clusters.
     *   <li>single: the minimum distance between all observations of the two clusters.
     *   <li>average: the average distance between all observations of the two clusters.
     * </ul>
     */
    Param<String> LINKAGE =
            new StringParam(
                    "linkage",
                    "Criterion for computing distance between two clusters.",
                    LINKAGE_WARD,
                    ParamValidators.inArray(
                            LINKAGE_WARD, LINKAGE_COMPLETE, LINKAGE_AVERAGE, LINKAGE_SINGLE));

    Param<Boolean> COMPUTE_FULL_TREE =
            new BooleanParam(
                    "computeFullTree",
                    "Whether computes the full tree after convergence.",
                    false,
                    ParamValidators.notNull());

    default Integer getNumClusters() {
        return get(NUM_CLUSTERS);
    }

    default T setNumClusters(Integer value) {
        return set(NUM_CLUSTERS, value);
    }

    default String getLinkage() {
        return get(LINKAGE);
    }

    default T setLinkage(String value) {
        return set(LINKAGE, value);
    }

    default Double getDistanceThreshold() {
        return get(DISTANCE_THRESHOLD);
    }

    default T setDistanceThreshold(Double value) {
        return set(DISTANCE_THRESHOLD, value);
    }

    default Boolean getComputeFullTree() {
        return get(COMPUTE_FULL_TREE);
    }

    default T setComputeFullTree(Boolean value) {
        return set(COMPUTE_FULL_TREE, value);
    }
}
