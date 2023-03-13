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

package org.apache.flink.ml.feature.minmaxscaler;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * An Estimator which implements the MinMaxScaler algorithm. This algorithm rescales feature values
 * to a common range [min, max] defined by user.
 *
 * <blockquote>
 *
 * $$ Rescaled(value) = \frac{value - E_{min}}{E_{max} - E_{min}} * (max - min) + min $$
 *
 * </blockquote>
 *
 * <p>For the case \(E_{max} == E_{min}\), \(Rescaled(value) = 0.5 * (max + min)\).
 *
 * <p>See https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization).
 */
public class MinMaxScaler
        implements Estimator<MinMaxScaler, MinMaxScalerModel>, MinMaxScalerParams<MinMaxScaler> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public MinMaxScaler() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public MinMaxScalerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        final String inputCol = getInputCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<DenseVector> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, DenseVector>)
                                        value -> ((Vector) value.getField(inputCol)).toDense());
        DataStream<DenseVector> minMaxValues =
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
        DataStream<MinMaxScalerModelData> modelData =
                DataStreamUtils.mapPartition(
                        minMaxValues,
                        new RichMapPartitionFunction<DenseVector, MinMaxScalerModelData>() {
                            @Override
                            public void mapPartition(
                                    Iterable<DenseVector> values,
                                    Collector<MinMaxScalerModelData> out) {
                                Iterator<DenseVector> iter = values.iterator();
                                DenseVector minVector = iter.next();
                                DenseVector maxVector = iter.next();
                                out.collect(new MinMaxScalerModelData(minVector, maxVector));
                            }
                        });

        MinMaxScalerModel model =
                new MinMaxScalerModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    /**
     * A stream operator to compute the min and max values in each partition of the input bounded
     * data stream.
     */
    public static class MinMaxReduceFunctionOperator extends AbstractStreamOperator<DenseVector>
            implements OneInputStreamOperator<DenseVector, DenseVector>, BoundedOneInput {
        private ListState<DenseVector> minState;
        private ListState<DenseVector> maxState;

        private DenseVector minVector;
        private DenseVector maxVector;

        @Override
        public void endInput() {
            if (minVector != null) {
                output.collect(new StreamRecord<>(minVector));
                output.collect(new StreamRecord<>(maxVector));
            }
        }

        @Override
        public void processElement(StreamRecord<DenseVector> streamRecord) {
            DenseVector currentValue = streamRecord.getValue();
            if (minVector == null) {
                int vecSize = currentValue.size();
                minVector = new DenseVector(vecSize);
                maxVector = new DenseVector(vecSize);
                System.arraycopy(currentValue.values, 0, minVector.values, 0, vecSize);
                System.arraycopy(currentValue.values, 0, maxVector.values, 0, vecSize);
            } else {
                Preconditions.checkArgument(
                        currentValue.size() == maxVector.size(),
                        "CurrentValue should has same size with maxVector.");
                for (int i = 0; i < currentValue.size(); ++i) {
                    minVector.values[i] = Math.min(minVector.values[i], currentValue.values[i]);
                    maxVector.values[i] = Math.max(maxVector.values[i], currentValue.values[i]);
                }
            }
        }

        @Override
        @SuppressWarnings("unchecked")
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            minState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "minState", TypeInformation.of(DenseVector.class)));
            maxState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "maxState", TypeInformation.of(DenseVector.class)));

            OperatorStateUtils.getUniqueElement(minState, "minState").ifPresent(x -> minVector = x);
            OperatorStateUtils.getUniqueElement(maxState, "maxState").ifPresent(x -> maxVector = x);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            minState.clear();
            maxState.clear();
            if (minVector != null) {
                minState.add(minVector);
                maxState.add(maxVector);
            }
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static MinMaxScaler load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
