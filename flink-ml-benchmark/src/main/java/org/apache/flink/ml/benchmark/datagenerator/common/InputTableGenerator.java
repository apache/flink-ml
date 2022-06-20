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

import org.apache.flink.ml.benchmark.datagenerator.InputDataGenerator;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;

import java.util.HashMap;
import java.util.Map;

/** Base class for generating data as input table arrays. */
public abstract class InputTableGenerator<T extends InputTableGenerator<T>>
        implements InputDataGenerator<T> {
    protected final Map<Param<?>, Object> paramMap = new HashMap<>();

    public InputTableGenerator() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public final Table[] getData(StreamTableEnvironment tEnv) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);

        RowGenerator[] rowGenerators = getRowGenerators();
        Table[] dataTables = new Table[rowGenerators.length];
        for (int i = 0; i < rowGenerators.length; i++) {
            DataStream<Row> dataStream =
                    env.addSource(rowGenerators[i], "sourceOp-" + i)
                            .returns(rowGenerators[i].getRowTypeInfo());
            dataTables[i] = tEnv.fromDataStream(dataStream);
        }

        return dataTables;
    }

    /** Gets generators for all input tables. */
    protected abstract RowGenerator[] getRowGenerators();

    @Override
    public final Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
