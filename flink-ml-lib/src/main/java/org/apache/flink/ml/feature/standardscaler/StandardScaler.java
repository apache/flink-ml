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

package org.apache.flink.ml.feature.standardscaler;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
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
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * An Estimator which implements the standard scaling algorithm.
 *
 * <p>Standardization is a common requirement for machine learning training because they may behave
 * badly if the individual features of an input do not look like standard normally distributed data
 * (e.g. Gaussian with 0 mean and unit variance).
 *
 * <p>This estimator standardizes the input features by removing the mean and scaling each dimension
 * to unit variance.
 */
public class StandardScaler
        implements Estimator<StandardScaler, StandardScalerModel>,
                StandardScalerParams<StandardScaler> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public StandardScaler() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public StandardScalerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Tuple3<DenseVector, DenseVector, Long>> sumAndSquaredSumAndWeight =
                tEnv.toDataStream(inputs[0])
                        .transform(
                                "computeMeta",
                                new TupleTypeInfo<>(
                                        TypeInformation.of(DenseVector.class),
                                        TypeInformation.of(DenseVector.class),
                                        BasicTypeInfo.LONG_TYPE_INFO),
                                new ComputeMetaOperator(getInputCol()));

        DataStream<StandardScalerModelData> modelData =
                sumAndSquaredSumAndWeight
                        .transform(
                                "buildModel",
                                TypeInformation.of(StandardScalerModelData.class),
                                new BuildModelOperator())
                        .setParallelism(1);

        StandardScalerModel model =
                new StandardScalerModel().setModelData(tEnv.fromDataStream(modelData));
        ReadWriteUtils.updateExistingParams(model, paramMap);
        return model;
    }

    /**
     * Builds the {@link StandardScalerModelData} using the meta data computed on each partition.
     */
    private static class BuildModelOperator extends AbstractStreamOperator<StandardScalerModelData>
            implements OneInputStreamOperator<
                            Tuple3<DenseVector, DenseVector, Long>, StandardScalerModelData>,
                    BoundedOneInput {
        private ListState<DenseVector> sumState;
        private ListState<DenseVector> squaredSumState;
        private ListState<Long> numElementsState;
        private DenseVector sum;
        private DenseVector squaredSum;
        private long numElements;

        @Override
        public void endInput() {
            if (numElements > 0) {
                BLAS.scal(1.0 / numElements, sum);
                double[] mean = sum.values;
                double[] std = squaredSum.values;
                if (numElements > 1) {
                    for (int i = 0; i < mean.length; i++) {
                        std[i] =
                                Math.sqrt(
                                        (squaredSum.values[i] - numElements * mean[i] * mean[i])
                                                / (numElements - 1));
                    }
                } else {
                    Arrays.fill(std, 0.0);
                }

                output.collect(
                        new StreamRecord<>(
                                new StandardScalerModelData(
                                        Vectors.dense(mean), Vectors.dense(std))));
            } else {
                throw new RuntimeException("The training set is empty.");
            }
        }

        @Override
        public void processElement(StreamRecord<Tuple3<DenseVector, DenseVector, Long>> element) {
            Tuple3<DenseVector, DenseVector, Long> value = element.getValue();
            if (numElements == 0) {
                sum = value.f0;
                squaredSum = value.f1;
                numElements = value.f2;
            } else {
                BLAS.axpy(1, value.f0, sum);
                BLAS.axpy(1, value.f1, squaredSum);
                numElements += value.f2;
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            sumState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "sumState", TypeInformation.of(DenseVector.class)));
            squaredSumState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "squaredSumState",
                                            TypeInformation.of(DenseVector.class)));
            numElementsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "numElementsState", BasicTypeInfo.LONG_TYPE_INFO));

            sum = OperatorStateUtils.getUniqueElement(sumState, "sumState").orElse(null);
            squaredSum =
                    OperatorStateUtils.getUniqueElement(squaredSumState, "squaredSumState")
                            .orElse(null);
            numElements =
                    OperatorStateUtils.getUniqueElement(numElementsState, "numElementsState")
                            .orElse(0L);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            if (numElements > 0) {
                sumState.update(Collections.singletonList(sum));
                squaredSumState.update(Collections.singletonList(squaredSum));
                numElementsState.update(Collections.singletonList(numElements));
            }
        }
    }

    /** Computes sum, squared sum and number of elements in each partition. */
    private static class ComputeMetaOperator
            extends AbstractStreamOperator<Tuple3<DenseVector, DenseVector, Long>>
            implements OneInputStreamOperator<Row, Tuple3<DenseVector, DenseVector, Long>>,
                    BoundedOneInput {
        private ListState<DenseVector> sumState;
        private ListState<DenseVector> squaredSumState;
        private ListState<Long> numElementsState;
        private DenseVector sum;
        private DenseVector squaredSum;
        private long numElements;

        private final String inputCol;

        public ComputeMetaOperator(String inputCol) {
            this.inputCol = inputCol;
        }

        @Override
        public void endInput() {
            if (numElements > 0) {
                output.collect(new StreamRecord<>(Tuple3.of(sum, squaredSum, numElements)));
            }
        }

        @Override
        public void processElement(StreamRecord<Row> element) {
            Vector inputVec = (Vector) element.getValue().getField(inputCol);
            if (numElements == 0) {
                sum = new DenseVector(inputVec.size());
                squaredSum = new DenseVector(inputVec.size());
            }
            BLAS.axpy(1, inputVec, sum);
            BLAS.hDot(inputVec, inputVec);
            BLAS.axpy(1, inputVec, squaredSum);
            numElements++;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            sumState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "sumState", TypeInformation.of(DenseVector.class)));
            squaredSumState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "squaredSumState",
                                            TypeInformation.of(DenseVector.class)));
            numElementsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "numElementsState", BasicTypeInfo.LONG_TYPE_INFO));

            sum = OperatorStateUtils.getUniqueElement(sumState, "sumState").orElse(null);
            squaredSum =
                    OperatorStateUtils.getUniqueElement(squaredSumState, "squaredSumState")
                            .orElse(null);
            numElements =
                    OperatorStateUtils.getUniqueElement(numElementsState, "numElementsState")
                            .orElse(0L);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            if (numElements > 0) {
                sumState.update(Collections.singletonList(sum));
                squaredSumState.update(Collections.singletonList(squaredSum));
                numElementsState.update(Collections.singletonList(numElements));
            }
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static StandardScaler load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
