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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * A Model which classifies data using the model data computed by {@link OnlineLogisticRegression}.
 */
public class OnlineLogisticRegressionModel
        implements Model<OnlineLogisticRegressionModel>,
                OnlineLogisticRegressionModelParams<OnlineLogisticRegressionModel> {
    public static final String MODEL_DATA_VERSION_GAUGE_KEY = "modelDataVersion";
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public OnlineLogisticRegressionModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                Types.DOUBLE,
                                TypeInformation.of(DenseIntDoubleVector.class),
                                Types.LONG),
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldNames(),
                                getPredictionCol(),
                                getRawPredictionCol(),
                                getModelVersionCol()));

        DataStream<Row> predictionResult =
                tEnv.toDataStream(inputs[0])
                        .connect(
                                LogisticRegressionModelDataUtil.getModelDataStream(modelDataTable)
                                        .broadcast())
                        .transform(
                                "PredictLabelOperator",
                                outputTypeInfo,
                                new PredictLabelOperator(inputTypeInfo, paramMap));

        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    /** A utility operator used for prediction. */
    private static class PredictLabelOperator extends AbstractStreamOperator<Row>
            implements TwoInputStreamOperator<Row, LogisticRegressionModelDataSegment, Row> {
        private final RowTypeInfo inputTypeInfo;

        private final Map<Param<?>, Object> params;
        private ListState<Row> bufferedPointsState;
        private DenseIntDoubleVector coefficient;
        private long modelDataVersion = 0L;
        private LogisticRegressionModelServable servable;

        public PredictLabelOperator(RowTypeInfo inputTypeInfo, Map<Param<?>, Object> params) {
            this.inputTypeInfo = inputTypeInfo;
            this.params = params;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            bufferedPointsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("bufferedPoints", inputTypeInfo));
        }

        @Override
        public void open() throws Exception {
            super.open();

            getRuntimeContext()
                    .getMetricGroup()
                    .gauge(
                            MODEL_DATA_VERSION_GAUGE_KEY,
                            (Gauge<String>) () -> Long.toString(modelDataVersion));
        }

        @Override
        public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
            processElement(streamRecord);
        }

        @Override
        public void processElement2(StreamRecord<LogisticRegressionModelDataSegment> streamRecord)
                throws Exception {
            LogisticRegressionModelDataSegment modelData = streamRecord.getValue();
            coefficient = modelData.coefficient;
            modelDataVersion = modelData.modelVersion;
            for (Row dataPoint : bufferedPointsState.get()) {
                processElement(new StreamRecord<>(dataPoint));
            }
            bufferedPointsState.clear();
        }

        public void processElement(StreamRecord<Row> streamRecord) throws Exception {
            Row dataPoint = streamRecord.getValue();
            if (coefficient == null) {
                bufferedPointsState.add(dataPoint);
                return;
            }
            if (servable == null) {
                servable =
                        new LogisticRegressionModelServable(
                                new LogisticRegressionModelDataSegment(coefficient, 0L));
                ParamUtils.updateExistingParams(servable, params);
            }
            IntDoubleVector features =
                    (IntDoubleVector) dataPoint.getField(servable.getFeaturesCol());
            Tuple2<Double, DenseIntDoubleVector> predictionResult = servable.transform(features);

            output.collect(
                    new StreamRecord<>(
                            Row.join(
                                    dataPoint,
                                    Row.of(
                                            predictionResult.f0,
                                            predictionResult.f1,
                                            modelDataVersion))));
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static OnlineLogisticRegressionModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public OnlineLogisticRegressionModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }
}
