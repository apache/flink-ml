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

package org.apache.flink.ml.api.core;

import org.apache.flink.api.common.state.BroadcastState;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Assert;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/** Defines a few Stage subclasses to be used in unit tests. */
public class ExampleStages {
    /**
     * A Model subclass that increments every value in the input stream by `delta` and outputs the
     * resulting values.
     */
    public static class SumModel implements Model<SumModel> {
        private final Map<Param<?>, Object> paramMap = new HashMap<>();
        private DataStream<Integer> modelData;

        // This empty constructor is necessary in order for ModelA to be loaded by
        // ReadWriteUtils.createStageWithParam
        public SumModel() {
            ParamUtils.initializeMapWithDefaultValues(paramMap, this);
        }

        @Override
        public Map<Param<?>, Object> getParamMap() {
            return paramMap;
        }

        @Override
        public Table[] transform(Table... inputs) {
            Assert.assertEquals(1, inputs.length);
            StreamTableEnvironment tEnv =
                    (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
            DataStream<Integer> input = tEnv.toDataStream(inputs[0], Integer.class);
            DataStream<Integer> output =
                    input.connect(modelData.broadcast())
                            .transform(
                                    "ApplyDeltaOperator",
                                    BasicTypeInfo.INT_TYPE_INFO,
                                    new ApplyDeltaOperator());

            return new Table[] {tEnv.fromDataStream(output)};
        }

        @Override
        public void setModelData(Table... inputs) {
            StreamTableEnvironment tEnv =
                    (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

            modelData = tEnv.toDataStream(inputs[0], Integer.class);
        }

        @Override
        public void save(String path) throws IOException {
            ReadWriteUtils.saveMetadata(this, path);

            File dataDir = new File(path, "data");
            if (!dataDir.mkdir()) {
                throw new IOException("Directory " + dataDir.toString() + " already exists.");
            }

            File dataFile = new File(dataDir, "delta");
            if (!dataFile.createNewFile()) {
                throw new IOException("File " + dataFile.toString() + " already exists.");
            }

            int delta;
            try {
                delta = (Integer) IteratorUtils.toList(modelData.executeAndCollect()).get(0);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            try (DataOutputStream outputStream =
                    new DataOutputStream(new FileOutputStream(dataFile))) {
                outputStream.writeInt(delta);
            }
        }

        public static SumModel load(String path) throws IOException {
            SumModel model = ReadWriteUtils.loadStageParam(path);
            File dataFile = Paths.get(path, "data", "delta").toFile();

            try (DataInputStream inputStream = new DataInputStream(new FileInputStream(dataFile))) {
                StreamExecutionEnvironment env =
                        StreamExecutionEnvironment.getExecutionEnvironment();
                StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
                Table modelData = tEnv.fromDataStream(env.fromElements(inputStream.readInt()));
                model.setModelData(modelData);
                return model;
            }
        }
    }

    // Adds delta from the 2nd input to every element in the 1st input and returns the added values.
    private static class ApplyDeltaOperator extends AbstractStreamOperator<Integer>
            implements TwoInputStreamOperator<Integer, Integer, Integer> {
        private ListState<Integer> unProcessedValues;
        private BroadcastState<String, Integer> broadcastState = null;

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            unProcessedValues =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<Integer>(
                                            "unProcessedValues", Integer.class));
            broadcastState =
                    context.getOperatorStateStore()
                            .getBroadcastState(
                                    new MapStateDescriptor<String, Integer>(
                                            "broadcastState", String.class, Integer.class));
        }

        @Override
        public void processElement1(StreamRecord<Integer> record) throws Exception {
            if (broadcastState.get("delta") == null) {
                unProcessedValues.add(record.getValue());
            } else {
                output.collect(new StreamRecord<>(record.getValue() + broadcastState.get("delta")));
            }
        }

        @Override
        public void processElement2(StreamRecord<Integer> record) throws Exception {
            if (broadcastState.get("delta") != null) {
                throw new IllegalStateException("Model data should have exactly one value");
            }
            broadcastState.put("delta", record.getValue());

            for (Integer value : unProcessedValues.get()) {
                output.collect(new StreamRecord<>(value + record.getValue()));
            }
            unProcessedValues.clear();
        }
    }

    /**
     * An Estimator subclass which calculates the sum of input values and instantiates a ModelA
     * instance with delta=sum(inputs).
     */
    public static class SumEstimator implements Estimator<SumEstimator, SumModel> {
        private final Map<Param<?>, Object> paramMap = new HashMap<>();

        public SumEstimator() {
            ParamUtils.initializeMapWithDefaultValues(paramMap, this);
        }

        @Override
        public Map<Param<?>, Object> getParamMap() {
            return paramMap;
        }

        @Override
        public SumModel fit(Table... inputs) {
            Assert.assertEquals(1, inputs.length);
            StreamTableEnvironment tEnv =
                    (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

            DataStream<Integer> input = tEnv.toDataStream(inputs[0], Integer.class);
            DataStream<Integer> modelData =
                    input.transform("SumOperator", BasicTypeInfo.INT_TYPE_INFO, new SumOperator())
                            .setParallelism(1);
            try {
                SumModel model = new SumModel();
                model.setModelData(tEnv.fromDataStream(modelData));

                return model;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void save(String path) throws IOException {
            ReadWriteUtils.saveMetadata(this, path);
        }

        public static SumEstimator load(String path) throws IOException {
            return ReadWriteUtils.loadStageParam(path);
        }
    }

    private static class SumOperator extends AbstractStreamOperator<Integer>
            implements OneInputStreamOperator<Integer, Integer>, BoundedOneInput {
        int sum = 0;

        @Override
        public void endInput() throws Exception {
            output.collect(new StreamRecord<>(sum));
        }

        @Override
        public void processElement(StreamRecord<Integer> input) throws Exception {
            sum += input.getValue();
        }
    }
}
