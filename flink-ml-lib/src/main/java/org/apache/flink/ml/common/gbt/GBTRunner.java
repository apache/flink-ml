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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifierParams;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.common.gbt.defs.TaskType;
import org.apache.flink.ml.common.gbt.defs.TrainContext;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.SparseVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressorParams;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** Runs a gradient boosting trees implementation. */
public class GBTRunner {

    public static DataStream<GBTModelData> trainClassifier(Table data, BaseGBTParams<?> estimator) {
        return train(data, estimator, TaskType.CLASSIFICATION);
    }

    public static DataStream<GBTModelData> trainRegressor(Table data, BaseGBTParams<?> estimator) {
        return train(data, estimator, TaskType.REGRESSION);
    }

    private static boolean isVectorType(TypeInformation<?> typeInfo) {
        return typeInfo instanceof DenseVectorTypeInfo
                || typeInfo instanceof SparseVectorTypeInfo
                || typeInfo instanceof VectorTypeInfo;
    }

    static DataStream<GBTModelData> train(
            Table data, BaseGBTParams<?> estimator, TaskType taskType) {
        String[] featuresCols = estimator.getFeaturesCols();
        TypeInformation<?>[] featuresTypes =
                Arrays.stream(featuresCols)
                        .map(d -> TableUtils.getTypeInfoByName(data.getResolvedSchema(), d))
                        .toArray(TypeInformation[]::new);
        for (int i = 0; i < featuresCols.length; i += 1) {
            Preconditions.checkArgument(
                    null != featuresTypes[i],
                    String.format(
                            "Column name %s not existed in the input data.", featuresCols[i]));
        }

        boolean isInputVector = featuresCols.length == 1 && isVectorType(featuresTypes[0]);
        return train(data, fromEstimator(estimator, isInputVector, taskType));
    }

    /** Trains a gradient boosting tree model with given data and parameters. */
    static DataStream<GBTModelData> train(Table dataTable, GbtParams p) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();
        Tuple2<Table, DataStream<FeatureMeta>> preprocessResult =
                p.isInputVector
                        ? Preprocess.preprocessVecCol(dataTable, p)
                        : Preprocess.preprocessCols(dataTable, p);
        dataTable = preprocessResult.f0;
        DataStream<FeatureMeta> featureMeta = preprocessResult.f1;

        DataStream<Row> data = tEnv.toDataStream(dataTable);
        DataStream<Tuple2<Double, Long>> labelSumCount =
                DataStreamUtils.aggregate(data, new LabelSumCountFunction(p.labelCol));
        return boost(dataTable, p, featureMeta, labelSumCount);
    }

    private static DataStream<GBTModelData> boost(
            Table dataTable,
            GbtParams p,
            DataStream<FeatureMeta> featureMeta,
            DataStream<Tuple2<Double, Long>> labelSumCount) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();

        final String featureMetaBcName = "featureMeta";
        final String labelSumCountBcName = "labelSumCount";
        Map<String, DataStream<?>> bcMap = new HashMap<>();
        bcMap.put(featureMetaBcName, featureMeta);
        bcMap.put(labelSumCountBcName, labelSumCount);

        DataStream<TrainContext> initTrainContext =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(
                                tEnv.toDataStream(tEnv.fromValues(0), Integer.class)),
                        bcMap,
                        (inputs) -> {
                            //noinspection unchecked
                            DataStream<Integer> input = (DataStream<Integer>) (inputs.get(0));
                            return input.map(
                                    new InitTrainContextFunction(
                                            featureMetaBcName, labelSumCountBcName, p));
                        });

        DataStream<Row> data = tEnv.toDataStream(dataTable);
        DataStreamList dataStreamList =
                Iterations.iterateBoundedStreamsUntilTermination(
                        DataStreamList.of(initTrainContext.broadcast()),
                        ReplayableDataStreamList.notReplay(data, featureMeta),
                        IterationConfig.newBuilder()
                                .setOperatorLifeCycle(IterationConfig.OperatorLifeCycle.ALL_ROUND)
                                .build(),
                        new BoostIterationBody(p));
        return dataStreamList.get(0);
    }

    public static GbtParams fromEstimator(
            BaseGBTParams<?> estimator, boolean isInputVector, TaskType taskType) {
        final Map<Param<?>, Object> paramMap = estimator.getParamMap();
        final Set<Param<?>> unsupported =
                new HashSet<>(
                        Arrays.asList(
                                BaseGBTParams.WEIGHT_COL,
                                BaseGBTParams.LEAF_COL,
                                BaseGBTParams.VALIDATION_INDICATOR_COL));
        List<Param<?>> unsupportedButSet =
                unsupported.stream()
                        .filter(d -> null != paramMap.get(d))
                        .collect(Collectors.toList());
        if (!unsupportedButSet.isEmpty()) {
            throw new UnsupportedOperationException(
                    String.format(
                            "Parameters %s are not supported yet right now.",
                            unsupportedButSet.stream()
                                    .map(d -> d.name)
                                    .collect(Collectors.joining(", "))));
        }

        GbtParams p = new GbtParams();
        p.taskType = taskType;
        p.featuresCols = estimator.getFeaturesCols();
        p.isInputVector = isInputVector;
        p.labelCol = estimator.getLabelCol();
        p.weightCol = estimator.getWeightCol();
        p.categoricalCols = estimator.getCategoricalCols();

        p.maxDepth = estimator.getMaxDepth();
        p.maxBins = estimator.getMaxBins();
        p.minInstancesPerNode = estimator.getMinInstancesPerNode();
        p.minWeightFractionPerNode = estimator.getMinWeightFractionPerNode();
        p.minInfoGain = estimator.getMinInfoGain();
        p.maxIter = estimator.getMaxIter();
        p.stepSize = estimator.getStepSize();
        p.seed = estimator.getSeed();
        p.subsamplingRate = estimator.getSubsamplingRate();
        p.featureSubsetStrategy = estimator.getFeatureSubsetStrategy();
        p.validationTol = estimator.getValidationTol();
        p.gamma = estimator.getRegGamma();
        p.lambda = estimator.getRegLambda();

        if (TaskType.CLASSIFICATION.equals(p.taskType)) {
            p.lossType = estimator.get(GBTClassifierParams.LOSS_TYPE);
        } else {
            p.lossType = estimator.get(GBTRegressorParams.LOSS_TYPE);
        }
        p.maxNumLeaves = 1 << p.maxDepth - 1;
        p.useMissing = true;
        return p;
    }

    private static class InitTrainContextFunction extends RichMapFunction<Integer, TrainContext> {
        private final String featureMetaBcName;
        private final String labelSumCountBcName;
        private final GbtParams p;

        private InitTrainContextFunction(
                String featureMetaBcName, String labelSumCountBcName, GbtParams p) {
            this.featureMetaBcName = featureMetaBcName;
            this.labelSumCountBcName = labelSumCountBcName;
            this.p = p;
        }

        @Override
        public TrainContext map(Integer value) {
            TrainContext trainContext = new TrainContext();
            trainContext.params = p;
            trainContext.featureMetas =
                    getRuntimeContext()
                            .<FeatureMeta>getBroadcastVariable(featureMetaBcName)
                            .toArray(new FeatureMeta[0]);
            if (!trainContext.params.isInputVector) {
                Arrays.sort(
                        trainContext.featureMetas,
                        Comparator.comparing(d -> ArrayUtils.indexOf(p.featuresCols, d.name)));
            }
            trainContext.numFeatures = trainContext.featureMetas.length;
            trainContext.labelSumCount =
                    getRuntimeContext()
                            .<Tuple2<Double, Long>>getBroadcastVariable(labelSumCountBcName)
                            .get(0);
            return trainContext;
        }
    }

    private static class LabelSumCountFunction
            implements AggregateFunction<Row, Tuple2<Double, Long>, Tuple2<Double, Long>> {

        private final String labelCol;

        private LabelSumCountFunction(String labelCol) {
            this.labelCol = labelCol;
        }

        @Override
        public Tuple2<Double, Long> createAccumulator() {
            return Tuple2.of(0., 0L);
        }

        @Override
        public Tuple2<Double, Long> add(Row value, Tuple2<Double, Long> accumulator) {
            double label = ((Number) value.getFieldAs(labelCol)).doubleValue();
            return Tuple2.of(accumulator.f0 + label, accumulator.f1 + 1);
        }

        @Override
        public Tuple2<Double, Long> getResult(Tuple2<Double, Long> accumulator) {
            return accumulator;
        }

        @Override
        public Tuple2<Double, Long> merge(Tuple2<Double, Long> a, Tuple2<Double, Long> b) {
            return Tuple2.of(a.f0 + b.f0, a.f1 + b.f1);
        }
    }
}