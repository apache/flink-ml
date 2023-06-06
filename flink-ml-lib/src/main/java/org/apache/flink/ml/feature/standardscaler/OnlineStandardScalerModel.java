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
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.metrics.MLMetrics;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamElementSerializer;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** A Model which transforms data using the model data computed by {@link OnlineStandardScaler}. */
public class OnlineStandardScalerModel
        implements Model<OnlineStandardScalerModel>,
                OnlineStandardScalerModelParams<OnlineStandardScalerModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public OnlineStandardScalerModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        String modelVersionCol = getModelVersionCol();

        TypeInformation<?>[] outputTypes;
        String[] outputNames;
        if (modelVersionCol == null) {
            outputTypes = ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE);
            outputNames = ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getOutputCol());
        } else {
            outputTypes =
                    ArrayUtils.addAll(
                            inputTypeInfo.getFieldTypes(), VectorTypeInfo.INSTANCE, Types.LONG);
            outputNames =
                    ArrayUtils.addAll(
                            inputTypeInfo.getFieldNames(), getOutputCol(), modelVersionCol);
        }
        RowTypeInfo outputTypeInfo = new RowTypeInfo(outputTypes, outputNames);

        DataStream<Row> predictionResult =
                tEnv.toDataStream(inputs[0])
                        .connect(
                                StandardScalerModelData.getModelDataStream(modelDataTable)
                                        .broadcast())
                        .transform(
                                "PredictionOperator",
                                outputTypeInfo,
                                new PredictionOperator(
                                        inputTypeInfo,
                                        getInputCol(),
                                        getWithMean(),
                                        getWithStd(),
                                        getMaxAllowedModelDelayMs(),
                                        getModelVersionCol()));

        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    /** A utility operator used for prediction. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private static class PredictionOperator extends AbstractStreamOperator<Row>
            implements TwoInputStreamOperator<Row, StandardScalerModelData, Row> {
        private final RowTypeInfo inputTypeInfo;

        private final String inputCol;

        private final boolean withMean;

        private final boolean withStd;

        private final long maxAllowedModelDelayMs;

        private final String modelVersionCol;

        private ListState<StreamRecord> bufferedPointsState;

        private ListState<StandardScalerModelData> modelDataState;

        /** Model data for inference. */
        private StandardScalerModelData modelData;

        private DenseIntDoubleVector mean;

        /** Inverse of standard deviation. */
        private DenseIntDoubleVector scale;

        private long modelVersion;

        private long modelTimeStamp;

        public PredictionOperator(
                RowTypeInfo inputTypeInfo,
                String inputCol,
                boolean withMean,
                boolean withStd,
                long maxAllowedModelDelayMs,
                String modelVersionCol) {
            this.inputTypeInfo = inputTypeInfo;
            this.inputCol = inputCol;
            this.withMean = withMean;
            this.withStd = withStd;
            this.maxAllowedModelDelayMs = maxAllowedModelDelayMs;
            this.modelVersionCol = modelVersionCol;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            bufferedPointsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<StreamRecord>(
                                            "bufferedPoints",
                                            new StreamElementSerializer(
                                                    inputTypeInfo.createSerializer(
                                                            getExecutionConfig()))));

            modelDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "modelData",
                                            TypeInformation.of(StandardScalerModelData.class)));
            modelData =
                    OperatorStateUtils.getUniqueElement(modelDataState, "modelData").orElse(null);
            if (modelData != null) {
                initializeModelData(modelData);
            } else {
                modelTimeStamp = -1;
                modelVersion = -1;
            }
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            if (modelData != null) {
                modelDataState.clear();
                modelDataState.add(modelData);
            }
        }

        @Override
        public void open() throws Exception {
            super.open();
            MetricGroup mlModelMetricGroup =
                    getRuntimeContext()
                            .getMetricGroup()
                            .addGroup(MLMetrics.ML_GROUP)
                            .addGroup(
                                    MLMetrics.ML_MODEL_GROUP,
                                    OnlineStandardScalerModel.class.getSimpleName());
            mlModelMetricGroup.gauge(MLMetrics.TIMESTAMP, (Gauge<Long>) () -> modelTimeStamp);
            mlModelMetricGroup.gauge(MLMetrics.VERSION, (Gauge<Long>) () -> modelVersion);
        }

        @Override
        public void processElement1(StreamRecord<Row> dataPoint) throws Exception {
            if (dataPoint.getTimestamp() - maxAllowedModelDelayMs <= modelTimeStamp
                    && mean != null) {
                doPrediction(dataPoint);
            } else {
                bufferedPointsState.add(dataPoint);
            }
        }

        @Override
        public void processElement2(StreamRecord<StandardScalerModelData> streamRecord)
                throws Exception {
            modelData = streamRecord.getValue();
            initializeModelData(modelData);

            // Does prediction on the cached data.
            List<StreamRecord> unprocessedElements = new ArrayList<>();
            boolean predictedCachedData = false;
            for (StreamRecord dataPoint : bufferedPointsState.get()) {
                if (dataPoint.getTimestamp() - maxAllowedModelDelayMs <= modelTimeStamp) {
                    doPrediction(dataPoint);
                    predictedCachedData = true;
                } else {
                    unprocessedElements.add(dataPoint);
                }
            }
            if (predictedCachedData) {
                bufferedPointsState.clear();
                if (unprocessedElements.size() > 0) {
                    bufferedPointsState.update(unprocessedElements);
                }
            }
        }

        private void initializeModelData(StandardScalerModelData modelData) {
            modelTimeStamp = modelData.timestamp;
            modelVersion = modelData.version;
            mean = modelData.mean;
            DenseIntDoubleVector std = modelData.std;

            if (withStd) {
                scale = std;
                double[] scaleValues = scale.values;
                for (int i = 0; i < scaleValues.length; i++) {
                    scaleValues[i] = scaleValues[i] == 0 ? 0 : 1 / scaleValues[i];
                }
            }
        }

        private void doPrediction(StreamRecord<Row> streamRecord) {
            Row dataPoint = streamRecord.getValue();

            IntDoubleVector outputVec =
                    ((IntDoubleVector) (Objects.requireNonNull(dataPoint.getField(inputCol))))
                            .clone();
            if (withMean) {
                outputVec = outputVec.toDense();
                BLAS.axpy(-1, mean, (DenseIntDoubleVector) outputVec);
            }
            if (withStd) {
                BLAS.hDot(scale, outputVec);
            }

            if (modelVersionCol == null) {
                output.collect(
                        new StreamRecord<>(
                                Row.join(dataPoint, Row.of(outputVec)),
                                streamRecord.getTimestamp()));
            } else {
                output.collect(
                        new StreamRecord<>(
                                Row.join(dataPoint, Row.of(outputVec, modelVersion)),
                                streamRecord.getTimestamp()));
            }
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static OnlineStandardScalerModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public OnlineStandardScalerModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }
}
