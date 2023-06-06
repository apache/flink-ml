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

package org.apache.flink.ml.feature.maxabsscaler;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
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
import java.util.HashMap;
import java.util.Map;

/**
 * An Estimator which implements the MaxAbsScaler algorithm. This algorithm rescales feature values
 * to the range [-1, 1] by dividing through the largest maximum absolute value in each feature. It
 * does not shift/center the data and thus does not destroy any sparsity.
 */
public class MaxAbsScaler
        implements Estimator<MaxAbsScaler, MaxAbsScalerModel>, MaxAbsScalerParams<MaxAbsScaler> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public MaxAbsScaler() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public MaxAbsScalerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        final String inputCol = getInputCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<Vector> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, Vector>)
                                        value -> ((Vector) value.getField(inputCol)),
                                VectorTypeInfo.INSTANCE);

        DataStream<Vector> maxAbsValues =
                inputData
                        .transform(
                                "reduceInEachPartition",
                                VectorTypeInfo.INSTANCE,
                                new MaxAbsReduceFunctionOperator())
                        .transform(
                                "reduceInFinalPartition",
                                VectorTypeInfo.INSTANCE,
                                new MaxAbsReduceFunctionOperator())
                        .setParallelism(1);

        DataStream<MaxAbsScalerModelData> modelData =
                maxAbsValues.map(
                        (MapFunction<Vector, MaxAbsScalerModelData>)
                                vector -> new MaxAbsScalerModelData((DenseIntDoubleVector) vector));

        MaxAbsScalerModel model =
                new MaxAbsScalerModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    /**
     * A stream operator to compute the maximum absolute values in each partition of the input
     * bounded data stream.
     */
    private static class MaxAbsReduceFunctionOperator extends AbstractStreamOperator<Vector>
            implements OneInputStreamOperator<Vector, Vector>, BoundedOneInput {
        private ListState<DenseIntDoubleVector> maxAbsState;
        private DenseIntDoubleVector maxAbsVector;

        @Override
        public void endInput() {
            if (maxAbsVector != null) {
                output.collect(new StreamRecord<>(maxAbsVector));
            }
        }

        @Override
        public void processElement(StreamRecord<Vector> streamRecord) {
            IntDoubleVector currentValue = (IntDoubleVector) streamRecord.getValue();

            maxAbsVector =
                    maxAbsVector == null
                            ? new DenseIntDoubleVector(currentValue.size())
                            : maxAbsVector;
            Preconditions.checkArgument(
                    currentValue.size() == maxAbsVector.size(),
                    "The training data should all have same dimensions.");

            if (currentValue instanceof DenseIntDoubleVector) {
                double[] values = ((DenseIntDoubleVector) currentValue).values;
                for (int i = 0; i < currentValue.size(); ++i) {
                    maxAbsVector.values[i] = Math.max(maxAbsVector.values[i], Math.abs(values[i]));
                }
            } else if (currentValue instanceof SparseIntDoubleVector) {
                int[] indices = ((SparseIntDoubleVector) currentValue).indices;
                double[] values = ((SparseIntDoubleVector) currentValue).values;

                for (int i = 0; i < indices.length; ++i) {
                    maxAbsVector.values[indices[i]] =
                            Math.max(maxAbsVector.values[indices[i]], Math.abs(values[i]));
                }
            }
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            maxAbsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "maxAbsState", DenseIntDoubleVectorTypeInfo.INSTANCE));

            OperatorStateUtils.getUniqueElement(maxAbsState, "maxAbsState")
                    .ifPresent(x -> maxAbsVector = x);
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            maxAbsState.clear();
            if (maxAbsVector != null) {
                maxAbsState.add(maxAbsVector);
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

    public static MaxAbsScaler load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
