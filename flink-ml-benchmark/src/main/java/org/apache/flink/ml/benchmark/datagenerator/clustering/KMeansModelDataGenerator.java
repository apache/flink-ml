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

import org.apache.flink.ml.benchmark.datagenerator.DataGenerator;
import org.apache.flink.ml.benchmark.datagenerator.InputDataGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.DenseVectorArrayGenerator;
import org.apache.flink.ml.benchmark.datagenerator.param.HasArraySize;
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.catalog.DataTypeFactory;
import org.apache.flink.table.functions.ScalarFunction;
import org.apache.flink.table.types.inference.TypeInference;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.apache.flink.table.api.Expressions.$;
import static org.apache.flink.table.api.Expressions.call;

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
        ParamUtils.updateExistingParams(vectorArrayGenerator, paramMap);
        vectorArrayGenerator.setNumValues(1);
        vectorArrayGenerator.setColNames(new String[] {"centroids"});

        Table centroidsTable = vectorArrayGenerator.getData(tEnv)[0];

        Table modelDataTable =
                centroidsTable.select(
                        $("centroids"),
                        call(GenerateWeightsFunction.class, $("centroids")).as("weights"));

        return new Table[] {modelDataTable};
    }

    /**
     * A scalar function that generates the weights vector for KMeansModelData from the centroids
     * information.
     */
    public static class GenerateWeightsFunction extends ScalarFunction {
        public DenseVector eval(DenseVector[] centroids) {
            return new DenseVector(centroids.length);
        }

        @Override
        public TypeInference getTypeInference(DataTypeFactory typeFactory) {
            return TypeInference.newBuilder()
                    .outputTypeStrategy(
                            callContext ->
                                    Optional.of(
                                            DataTypes.of(DenseVectorTypeInfo.INSTANCE)
                                                    .toDataType(typeFactory)))
                    .build();
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
