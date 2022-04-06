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

package org.apache.flink.ml.benchmark.datagenerator.clustering;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.ml.benchmark.datagenerator.DataGenerator;
import org.apache.flink.ml.benchmark.datagenerator.InputDataGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.DenseVectorArrayGenerator;
import org.apache.flink.ml.benchmark.datagenerator.param.HasArraySize;
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.clustering.kmeans.KMeansModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

import java.util.HashMap;
import java.util.Map;

/**
 * A DataGenerator which creates a table containing one {@link
 * org.apache.flink.ml.clustering.kmeans.KMeansModel} instance.
 */
public class KMeansModelDataGenerator
        implements DataGenerator<KMeansModelDataGenerator>,
                HasVectorDim<KMeansModelDataGenerator>,
                HasArraySize<KMeansModelDataGenerator> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public KMeansModelDataGenerator() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] getData(StreamTableEnvironment tEnv) {
        InputDataGenerator<?> vectorArrayGenerator = new DenseVectorArrayGenerator();
        ReadWriteUtils.updateExistingParams(vectorArrayGenerator, paramMap);
        vectorArrayGenerator.setNumValues(1);

        Table vectorArrayTable = vectorArrayGenerator.getData(tEnv)[0];
        DataStream<KMeansModelData> modelDataStream =
                tEnv.toDataStream(vectorArrayTable, DenseVector[].class)
                        .map(new GenerateKMeansModelDataFunction());

        return new Table[] {tEnv.fromDataStream(modelDataStream)};
    }

    private static class GenerateKMeansModelDataFunction
            implements MapFunction<DenseVector[], KMeansModelData> {
        @Override
        public KMeansModelData map(DenseVector[] vectors) {
            return new KMeansModelData(vectors, new DenseVector(vectors.length));
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
