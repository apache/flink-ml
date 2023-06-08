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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the online logistic regression algorithm. The online optimizer of
 * this algorithm is The FTRL-Proximal proposed by H.Brendan McMahan et al.
 *
 * <p>See <a href="https://doi.org/10.1145/2487575.2488200">H. Brendan McMahan et al., Ad click
 * prediction: a view from the trenches.</a>
 */
public class OnlineLogisticRegression
        implements Estimator<OnlineLogisticRegression, OnlineLogisticRegressionModel>,
                OnlineLogisticRegressionParams<OnlineLogisticRegression> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table initModelDataTable;

    public OnlineLogisticRegression() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OnlineLogisticRegressionModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<LogisticRegressionModelDataSegment> modelDataStream =
                LogisticRegressionModelDataUtil.getModelDataStream(initModelDataTable);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        TypeInformation pointTypeInfo;

        if (getWeightCol() == null) {
            pointTypeInfo =
                    Types.ROW(
                            inputTypeInfo.getTypeAt(getFeaturesCol()),
                            inputTypeInfo.getTypeAt(getLabelCol()));
        } else {
            pointTypeInfo =
                    Types.ROW(
                            inputTypeInfo.getTypeAt(getFeaturesCol()),
                            inputTypeInfo.getTypeAt(getLabelCol()),
                            inputTypeInfo.getTypeAt(getWeightCol()));
        }

        DataStream<Row> points =
                tEnv.toDataStream(inputs[0])
                        .map(
                                new FeaturesLabelExtractor(
                                        getFeaturesCol(), getLabelCol(), getWeightCol()),
                                pointTypeInfo);

        DataStream<DenseIntDoubleVector> initModelData =
                modelDataStream.map(
                        (MapFunction<LogisticRegressionModelDataSegment, DenseIntDoubleVector>)
                                value -> value.coefficient);

        initModelData.getTransformation().setParallelism(1);

        IterationBody body =
                new FtrlIterationBody(
                        getGlobalBatchSize(), getAlpha(), getBeta(), getReg(), getElasticNet());

        DataStream<LogisticRegressionModelDataSegment> onlineModelData =
                Iterations.iterateUnboundedStreams(
                                DataStreamList.of(initModelData), DataStreamList.of(points), body)
                        .get(0);

        Table onlineModelDataTable = tEnv.fromDataStream(onlineModelData);
        OnlineLogisticRegressionModel model =
                new OnlineLogisticRegressionModel().setModelData(onlineModelDataTable);
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    private static class FeaturesLabelExtractor implements MapFunction<Row, Row> {
        private final String featuresCol;
        private final String labelCol;
        private final String weightCol;

        private FeaturesLabelExtractor(String featuresCol, String labelCol, String weightCol) {
            this.featuresCol = featuresCol;
            this.labelCol = labelCol;
            this.weightCol = weightCol;
        }

        @Override
        public Row map(Row row) throws Exception {
            if (weightCol == null) {
                return Row.of(row.getField(featuresCol), row.getField(labelCol));
            } else {
                return Row.of(
                        row.getField(featuresCol), row.getField(labelCol), row.getField(weightCol));
            }
        }
    }

    /**
     * In the implementation of ftrl optimizer, gradients are calculated in distributed workers and
     * reduce them to one final gradient. The reduced gradient is used to update model by ftrl
     * method. When the feature vector is dense, it can get the same result as tensorflow's ftrl. If
     * feature vector is sparse, we use the mean value in every feature dim instead of mean value of
     * whole vector, which can get a better convergence.
     *
     * <p>See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl
     *
     * <p>todo: makes ftrl to be a common optimizer and place it in org.apache.flink.ml.common.
     */
    private static class FtrlIterationBody implements IterationBody {
        private final int batchSize;
        private final double alpha;
        private final double beta;
        private final double l1;
        private final double l2;

        public FtrlIterationBody(
                int batchSize, double alpha, double beta, double reg, double elasticNet) {
            this.batchSize = batchSize;
            this.alpha = alpha;
            this.beta = beta;
            this.l1 = elasticNet * reg;
            this.l2 = (1 - elasticNet) * reg;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<DenseIntDoubleVector> modelData = variableStreams.get(0);

            DataStream<Row> points = dataStreams.get(0);
            int parallelism = points.getParallelism();
            Preconditions.checkState(
                    parallelism <= batchSize,
                    "There are more subtasks in the training process than the number "
                            + "of elements in each batch. Some subtasks might be idling forever.");

            DataStream<DenseIntDoubleVector[]> newGradient =
                    DataStreamUtils.generateBatchData(points, parallelism, batchSize)
                            .connect(modelData.broadcast())
                            .transform(
                                    "LocalGradientCalculator",
                                    TypeInformation.of(DenseIntDoubleVector[].class),
                                    new CalculateLocalGradient())
                            .setParallelism(parallelism)
                            .countWindowAll(parallelism)
                            .reduce(
                                    (ReduceFunction<DenseIntDoubleVector[]>)
                                            (gradientInfo, newGradientInfo) -> {
                                                BLAS.axpy(1.0, gradientInfo[0], newGradientInfo[0]);
                                                BLAS.axpy(1.0, gradientInfo[1], newGradientInfo[1]);
                                                if (newGradientInfo[2] == null) {
                                                    newGradientInfo[2] = gradientInfo[2];
                                                }
                                                return newGradientInfo;
                                            });
            DataStream<DenseIntDoubleVector> feedbackModelData =
                    newGradient
                            .transform(
                                    "ModelDataUpdater",
                                    TypeInformation.of(DenseIntDoubleVector.class),
                                    new UpdateModel(alpha, beta, l1, l2))
                            .setParallelism(1);

            DataStream<LogisticRegressionModelDataSegment> outputModelData =
                    feedbackModelData.map(new CreateLrModelData()).setParallelism(1);
            return new IterationBodyResult(
                    DataStreamList.of(feedbackModelData), DataStreamList.of(outputModelData));
        }
    }

    private static class CreateLrModelData
            implements MapFunction<DenseIntDoubleVector, LogisticRegressionModelDataSegment>,
                    CheckpointedFunction {
        private Long modelVersion = 1L;
        private transient ListState<Long> modelVersionState;

        @Override
        public LogisticRegressionModelDataSegment map(DenseIntDoubleVector denseVector)
                throws Exception {
            return new LogisticRegressionModelDataSegment(denseVector, modelVersion++);
        }

        @Override
        public void snapshotState(FunctionSnapshotContext functionSnapshotContext)
                throws Exception {
            modelVersionState.update(Collections.singletonList(modelVersion));
        }

        @Override
        public void initializeState(FunctionInitializationContext context) throws Exception {
            modelVersionState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("modelVersionState", Long.class));
        }
    }

    /** Updates model. */
    private static class UpdateModel extends AbstractStreamOperator<DenseIntDoubleVector>
            implements OneInputStreamOperator<DenseIntDoubleVector[], DenseIntDoubleVector> {
        private ListState<double[]> nParamState;
        private ListState<double[]> zParamState;
        private final double alpha;
        private final double beta;
        private final double l1;
        private final double l2;
        private double[] nParam;
        private double[] zParam;

        public UpdateModel(double alpha, double beta, double l1, double l2) {
            this.alpha = alpha;
            this.beta = beta;
            this.l1 = l1;
            this.l2 = l2;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            nParamState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("nParamState", double[].class));
            zParamState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("zParamState", double[].class));
        }

        @Override
        public void processElement(StreamRecord<DenseIntDoubleVector[]> streamRecord)
                throws Exception {
            DenseIntDoubleVector[] gradientInfo = streamRecord.getValue();
            double[] coefficient = gradientInfo[2].values;
            double[] g = gradientInfo[0].values;
            for (int i = 0; i < g.length; ++i) {
                if (gradientInfo[1].values[i] != 0.0) {
                    g[i] = g[i] / gradientInfo[1].values[i];
                }
            }
            if (zParam == null) {
                zParam = new double[g.length];
                nParam = new double[g.length];
                nParamState.add(nParam);
                zParamState.add(zParam);
            }

            for (int i = 0; i < zParam.length; ++i) {
                double sigma = (Math.sqrt(nParam[i] + g[i] * g[i]) - Math.sqrt(nParam[i])) / alpha;
                zParam[i] += g[i] - sigma * coefficient[i];
                nParam[i] += g[i] * g[i];

                if (Math.abs(zParam[i]) <= l1) {
                    coefficient[i] = 0.0;
                } else {
                    coefficient[i] =
                            ((zParam[i] < 0 ? -1 : 1) * l1 - zParam[i])
                                    / ((beta + Math.sqrt(nParam[i])) / alpha + l2);
                }
            }
            output.collect(new StreamRecord<>(new DenseIntDoubleVector(coefficient)));
        }
    }

    private static class CalculateLocalGradient
            extends AbstractStreamOperator<DenseIntDoubleVector[]>
            implements TwoInputStreamOperator<Row[], DenseIntDoubleVector, DenseIntDoubleVector[]> {
        private ListState<DenseIntDoubleVector> modelDataState;
        private ListState<Row[]> localBatchDataState;
        private double[] gradient;
        private double[] weightSum;

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            modelDataState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "modelData", DenseIntDoubleVector.class));
            TypeInformation<Row[]> type =
                    ObjectArrayTypeInfo.getInfoFor(TypeInformation.of(Row.class));
            localBatchDataState =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("localBatch", type));
        }

        @Override
        public void processElement1(StreamRecord<Row[]> pointsRecord) throws Exception {
            localBatchDataState.add(pointsRecord.getValue());
            calculateGradient();
        }

        private void calculateGradient() throws Exception {
            if (!modelDataState.get().iterator().hasNext()
                    || !localBatchDataState.get().iterator().hasNext()) {
                return;
            }
            DenseIntDoubleVector modelData =
                    OperatorStateUtils.getUniqueElement(modelDataState, "modelData").get();
            modelDataState.clear();

            List<Row[]> pointsList = IteratorUtils.toList(localBatchDataState.get().iterator());
            Row[] points = pointsList.remove(0);
            localBatchDataState.update(pointsList);

            for (Row point : points) {
                IntDoubleVector vec = point.getFieldAs(0);
                double label = point.getFieldAs(1);
                double weight = point.getArity() == 2 ? 1.0 : point.getFieldAs(2);
                if (gradient == null) {
                    gradient = new double[vec.size()];
                    weightSum = new double[gradient.length];
                }
                double p = BLAS.dot(modelData, vec);
                p = 1 / (1 + Math.exp(-p));
                if (vec instanceof DenseIntDoubleVector) {
                    DenseIntDoubleVector dvec = (DenseIntDoubleVector) vec;
                    for (int i = 0; i < modelData.size(); ++i) {
                        gradient[i] += (p - label) * dvec.values[i];
                        weightSum[i] += 1.0;
                    }
                } else {
                    SparseIntDoubleVector svec = (SparseIntDoubleVector) vec;
                    for (int i = 0; i < svec.indices.length; ++i) {
                        int idx = svec.indices[i];
                        gradient[idx] += (p - label) * svec.values[i];
                        weightSum[idx] += weight;
                    }
                }
            }

            if (points.length > 0) {
                output.collect(
                        new StreamRecord<>(
                                new DenseIntDoubleVector[] {
                                    new DenseIntDoubleVector(gradient),
                                    new DenseIntDoubleVector(weightSum),
                                    (getRuntimeContext().getIndexOfThisSubtask() == 0)
                                            ? modelData
                                            : null
                                }));
            }
            Arrays.fill(gradient, 0.0);
            Arrays.fill(weightSum, 0.0);
        }

        @Override
        public void processElement2(StreamRecord<DenseIntDoubleVector> modelDataRecord)
                throws Exception {
            modelDataState.add(modelDataRecord.getValue());
            calculateGradient();
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                LogisticRegressionModelDataUtil.getModelDataStream(initModelDataTable),
                path,
                new LogisticRegressionModelDataUtil.ModelDataEncoder());
    }

    public static OnlineLogisticRegression load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        OnlineLogisticRegression onlineLogisticRegression = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv, path, new LogisticRegressionModelDataUtil.ModelDataDecoder());
        onlineLogisticRegression.setInitialModelData(modelDataTable);
        return onlineLogisticRegression;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Sets the initial model data of the online training process with the provided model data
     * table.
     */
    public OnlineLogisticRegression setInitialModelData(Table initModelDataTable) {
        this.initModelDataTable = initModelDataTable;
        return this;
    }
}
