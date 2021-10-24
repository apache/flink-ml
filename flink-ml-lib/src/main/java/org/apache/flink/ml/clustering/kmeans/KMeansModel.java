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

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.core.Model;
import org.apache.flink.ml.clustering.kmeans.KMeans.SelectNearestCentroid;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** A Model which clusters data into k clusters using the model data computed by {@link KMeans}. */
public class KMeansModel implements Model<KMeansModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private final DataSet<Tuple2<Integer, DenseVector>> centroids;

    public KMeansModel(DataSet<Tuple2<Integer, DenseVector>> centroids) {
        this.centroids = centroids;
    }

    @Override
    public Table[] transform(Table... inputs) {
        return new Table[0];
    }

    @Override
    public DataSet[] transformDataSet(DataSet... inputs) {
        DataSet<DenseVector> points = inputs[0];
        DataSet<Tuple2<Integer, DenseVector>> pointsWithCentroidId =
                points.map(new SelectNearestCentroid()).withBroadcastSet(centroids, "centroids");
        return new DataSet[] {pointsWithCentroidId};
    }

    @Override
    public void save(String path) throws IOException {
        // TODO: save model data.
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getUserDefinedParamMap() {
        return paramMap;
    }

    public static KMeansModel load(String path) throws IOException {
        // TODO: load model data.
        return ReadWriteUtils.loadStageParam(path);
    }
}
