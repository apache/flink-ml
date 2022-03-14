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
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.NumberSequenceIterator;
import org.apache.flink.util.Preconditions;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/** A DataGenerator which creates a table of DenseVector. */
public class DenseVectorGenerator
        implements InputDataGenerator<DenseVectorGenerator>, HasVectorDim<DenseVectorGenerator> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public DenseVectorGenerator() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] getData(StreamTableEnvironment tEnv) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        DataStream<DenseVector> dataStream =
                env.fromParallelCollection(
                                new NumberSequenceIterator(1L, getNumValues()),
                                BasicTypeInfo.LONG_TYPE_INFO)
                        .map(new RandomDenseVectorGenerator(getSeed(), getVectorDim()));

        Table dataTable = tEnv.fromDataStream(dataStream);
        if (getColNames() != null) {
            Preconditions.checkState(getColNames().length > 0);
            dataTable = dataTable.as(getColNames()[0]);
        }

        return new Table[] {dataTable};
    }

    private static class RandomDenseVectorGenerator extends RichMapFunction<Long, DenseVector> {
        private final int vectorDim;
        private final long initSeed;
        private Random random;

        private RandomDenseVectorGenerator(long initSeed, int vectorDim) {
            this.vectorDim = vectorDim;
            this.initSeed = initSeed;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            int index = getRuntimeContext().getIndexOfThisSubtask();
            random = new Random(Tuple2.of(initSeed, index).hashCode());
        }

        @Override
        public DenseVector map(Long value) {
            double[] values = new double[vectorDim];
            for (int i = 0; i < vectorDim; i++) {
                values[i] = random.nextDouble();
            }
            return Vectors.dense(values);
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
