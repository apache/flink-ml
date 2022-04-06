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

package org.apache.flink.ml.benchmark.datagenerator.common;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.benchmark.datagenerator.InputDataGenerator;
import org.apache.flink.ml.benchmark.datagenerator.param.HasArraySize;
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.NumberSequenceIterator;
import org.apache.flink.util.Preconditions;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/** A DataGenerator which creates a table of DenseVector array. */
public class DenseVectorArrayGenerator
        implements InputDataGenerator<DenseVectorArrayGenerator>,
                HasArraySize<DenseVectorArrayGenerator>,
                HasVectorDim<DenseVectorArrayGenerator> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public DenseVectorArrayGenerator() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] getData(StreamTableEnvironment tEnv) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        DataStream<DenseVector[]> dataStream =
                env.fromParallelCollection(
                                new NumberSequenceIterator(1L, getNumValues()),
                                BasicTypeInfo.LONG_TYPE_INFO)
                        .map(
                                new GenerateRandomContinuousVectorArrayFunction(
                                        getSeed(), getVectorDim(), getArraySize()));

        Schema schema = Schema.newBuilder().column("f0", DataTypes.of(DenseVector[].class)).build();
        Table dataTable = tEnv.fromDataStream(dataStream, schema);
        if (getColNames() != null) {
            Preconditions.checkState(getColNames().length > 0);
            dataTable = dataTable.as(getColNames()[0]);
        }

        return new Table[] {dataTable};
    }

    private static class GenerateRandomContinuousVectorArrayFunction
            extends RichMapFunction<Long, DenseVector[]> {
        private final int vectorDim;
        private final long initSeed;
        private final int arraySize;
        private Random random;

        private GenerateRandomContinuousVectorArrayFunction(
                long initSeed, int vectorDim, int arraySize) {
            this.vectorDim = vectorDim;
            this.initSeed = initSeed;
            this.arraySize = arraySize;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            int index = getRuntimeContext().getIndexOfThisSubtask();
            random = new Random(Tuple2.of(initSeed, index).hashCode());
        }

        @Override
        public DenseVector[] map(Long value) {
            DenseVector[] result = new DenseVector[arraySize];
            for (int i = 0; i < arraySize; i++) {
                result[i] = new DenseVector(vectorDim);
                for (int j = 0; j < vectorDim; j++) {
                    result[i].values[j] = random.nextDouble();
                }
            }
            return result;
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
