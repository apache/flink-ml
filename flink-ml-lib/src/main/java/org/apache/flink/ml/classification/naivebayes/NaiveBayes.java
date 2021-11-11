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

package org.apache.flink.ml.classification.naivebayes;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.ml.api.core.Estimator;
import org.apache.flink.ml.common.EndOfStreamWindows;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * Naive Bayes classifier is a simple probability classification algorithm using
 * Bayes theorem based on independent assumption. It is an independent feature model.
 * The input feature can be continual or categorical.
 */
public class NaiveBayes implements Estimator<NaiveBayes, NaiveBayesModel>,
        NaiveBayesParams<NaiveBayes> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    @Override
    public NaiveBayesModel fit(Table... inputs) {
        String[] featureColNames = getFeatureCols();
        String labelColName = getLabelCol();
        String predictionCol = getPredictionCol();
        double smoothing = getSmoothing();

        Preconditions.checkNotNull(inputs, "input table list should not be null");
        Preconditions.checkArgument(inputs.length == 1, "input table list should contain only one argument");
        Preconditions.checkArgument(
                new HashSet<>(Arrays.asList(featureColNames)).size() == featureColNames.length,
                "feature columns should not duplicate");
        Preconditions.checkNotNull(labelColName, "label column should be set");

        StreamTableEnvironment tEnv = (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);

        DataStream<NaiveBayesModelData> naiveBayesModel = input
                .flatMap(new FlattenFunction(
                        featureColNames,
                        labelColName
                ))
                .keyBy((KeySelector<Tuple4<Object, Integer, Object, Double>, Object>) value -> new Tuple3<>(value.f0, value.f1, value.f2))
                .window(EndOfStreamWindows.get())
                .reduce((ReduceFunction<Tuple4<Object, Integer, Object, Double>>) (t0, t1) -> {t0.f3 += t1.f3; return t0; })
                .keyBy((KeySelector<Tuple4<Object, Integer, Object, Double>, Object>) value -> new Tuple2<>(value.f0, value.f1))
                .window(EndOfStreamWindows.get())
                .aggregate(new ValueMapFunction())
                .keyBy((KeySelector<Tuple4<Object, Integer, Map<Object, Double>, Double>, Object>) value -> value.f0)
                .window(EndOfStreamWindows.get())
                .aggregate(new MapArrayFunction(featureColNames.length))
                .windowAll(EndOfStreamWindows.get())
                .apply(new GenerateModelFunction(
                        smoothing,
                        featureColNames));

        NaiveBayesModel model = new NaiveBayesModel()
                .setPredictionCol(predictionCol)
                .setFeatureCols(featureColNames);
        model.setModelData(
                tEnv.fromDataStream(naiveBayesModel)
        );
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static NaiveBayes load(String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Function to convert each column into tuples of label, feature column index, feature value, weight.
     */
    private static class FlattenFunction implements FlatMapFunction<Row, Tuple4<Object, Integer, Object, Double>> {
        private final String[] featureColNames;
        private final String labelColName;
        private final int featureSize;

        private FlattenFunction(String[] featureColNames, String labelColName) {
            this.labelColName = labelColName;
            this.featureColNames = featureColNames;
            this.featureSize = featureColNames.length;
        }

        @Override
        public void flatMap(Row row, Collector<Tuple4<Object, Integer, Object, Double>> collector) {
            Object label = row.getField(labelColName);
            if (label == null) {
                return;
            }

            for (int i = 0; i < featureSize; i++) {
                Object feature = row.getField(featureColNames[i]);
                if (feature == null) {
                    continue;
                }
                collector.collect(new Tuple4<>(label, i, feature, 1.0));
            }
        }
    }

    /**
     * Function to aggregate feature value and weight into map from records with the same label and feature column index.
     */
    private static class ValueMapFunction implements AggregateFunction<
            Tuple4<Object, Integer, Object, Double>,
            Tuple4<Object, Integer, Map<Object, Double>, Double>,
            Tuple4<Object, Integer, Map<Object, Double>, Double>> {

        @Override
        public Tuple4<Object, Integer, Map<Object, Double>, Double> createAccumulator() {
            return new Tuple4<>(new Object(), -1, new HashMap<>(), 0.);
        }

        @Override
        public Tuple4<Object, Integer, Map<Object, Double>, Double> add(
                Tuple4<Object, Integer, Object, Double> value,
                Tuple4<Object, Integer, Map<Object, Double>, Double> acc) {
            acc.f0 = value.f0;
            acc.f1 = value.f1;
            acc.f2.put(value.f2, value.f3);
            acc.f3 += value.f3;
            return acc;
        }

        @Override
        public Tuple4<Object, Integer, Map<Object, Double>, Double> getResult(
                Tuple4<Object, Integer, Map<Object, Double>, Double> acc) {
            return acc;
        }

        @Override
        public Tuple4<Object, Integer, Map<Object, Double>, Double> merge(
                Tuple4<Object, Integer, Map<Object, Double>, Double> acc0,
                Tuple4<Object, Integer, Map<Object, Double>, Double> acc1) {
            if (acc0.f0.equals(new Object())) {
                acc0.f0 = acc1.f0;
            }
            if (acc0.f1 == -1) {
                acc0.f1 = acc1.f1;
            }
            for (Map.Entry<Object, Double> entry: acc1.f2.entrySet()) {
                acc0.f2.put(entry.getKey(), entry.getValue());
            }
            acc0.f3 += acc1.f3;

            return acc0;
        }
    }

    /**
     * Function to aggregate maps under the same label into arrays. array len = featureSize
     */
    private static class MapArrayFunction implements AggregateFunction<
            Tuple4<Object, Integer, Map<Object, Double>, Double>,
            Tuple3<Object, Double[], Map <Object, Double>[]>,
            Tuple3<Object, Double[], Map <Object, Double>[]>> {
        private final int featureSize;

        private MapArrayFunction(int featureSize) {
            this.featureSize = featureSize;
        }

        @Override
        public Tuple3<Object, Double[], Map<Object, Double>[]> createAccumulator() {
            Double[] weightSum = new Double[featureSize];
            Arrays.fill(weightSum, 0.);
            return new Tuple3<>(new Object(), weightSum, new HashMap[featureSize]);
        }

        @Override
        public Tuple3<Object, Double[], Map<Object, Double>[]> add(
                Tuple4<Object, Integer, Map<Object, Double>, Double> value,
                Tuple3<Object, Double[], Map<Object, Double>[]> acc) {
            acc.f0 = value.f0;
            acc.f1[value.f1] = value.f3;
            acc.f2[value.f1] = value.f2;

            return acc;
        }

        @Override
        public Tuple3<Object, Double[], Map<Object, Double>[]> getResult(Tuple3<Object, Double[], Map<Object, Double>[]> acc) {
            return acc;
        }

        @Override
        public Tuple3<Object, Double[], Map<Object, Double>[]> merge(
                Tuple3<Object, Double[], Map<Object, Double>[]> acc0,
                Tuple3<Object, Double[], Map<Object, Double>[]> acc1) {
            if (acc0.f0.equals(new Object())) {
                acc0.f0 = acc1.f0;
            }
            for (int i = 0; i < featureSize; i++) {
                acc0.f1[i] += acc1.f1[i];
                if (acc1.f2[i] != null) {
                    acc0.f2[i] = acc1.f2[i];
                }
            }

            return acc0;
        }
    }

    /**
     * Function to generate Naive Bayes model data.
     */
    private static class GenerateModelFunction implements AllWindowFunction<
            Tuple3<Object, Double[], Map<Object, Double>[]>,
            NaiveBayesModelData,
            TimeWindow
            > {
        private final int featureSize;
        private final double smoothing;
        private final String[] featureColNames;

        GenerateModelFunction(double smoothing, String[] featureColNames) {
            this.smoothing = smoothing;
            this.featureColNames = featureColNames;
            this.featureSize = featureColNames.length;
        }

        @Override
        public void apply(TimeWindow timeWindow,
                          Iterable<Tuple3<Object, Double[], Map<Object, Double>[]>> values,
                          Collector<NaiveBayesModelData> out) {
            double[] numDocs = new double[featureSize];
            ArrayList <Tuple3 <Object, Double[], Map <Object, Double>[]>> modelArray = new ArrayList <>();
            HashSet <Object>[] categoryNumbers = new HashSet[featureSize];
            for (int i = 0; i < featureSize; i++) {
                categoryNumbers[i] = new HashSet <>();
            }
            for (Tuple3 <Object, Double[], Map <Object, Double>[]> tup : values) {
                modelArray.add(tup);
                for (int i = 0; i < featureSize; i++) {
                    numDocs[i] += tup.f1[i];
                    categoryNumbers[i].addAll(tup.f2[i].keySet());
                }
            }

            int[] categoryNumber = new int[featureSize];
            double piLog = 0;
            int numLabels = modelArray.size();
            for (int i = 0; i < featureSize; i++) {
                categoryNumber[i] = categoryNumbers[i].size();
                piLog += numDocs[i];
            }
            piLog = Math.log(piLog + numLabels * smoothing);

            Map<Object, Double>[][] theta = new HashMap[numLabels][featureSize];
            double[] piArray = new double[numLabels];
            Object[] labels = new Object[numLabels];

            // consider smoothing.
            for (int i = 0; i < numLabels; i++) {
                Map <Object, Double>[] param = modelArray.get(i).f2;
                for (int j = 0; j < featureSize; j++) {
                    Map<Object, Double> squareData = new HashMap<>();
                    double thetaLog = Math.log(modelArray.get(i).f1[j] + smoothing * categoryNumber[j]);
                    for (Object cate: categoryNumbers[j]) {
                        double value = 0.0;
                        if (param[j].containsKey(cate)) {
                            value = param[j].get(cate);
                        }
                        squareData.put(cate, Math.log(value + smoothing) - thetaLog);
                    }
                    theta[i][j] = squareData;
                }

                labels[i] = modelArray.get(i).f0;
                double weightSum = 0;
                for (Double weight : modelArray.get(i).f1) {
                    weightSum += weight;
                }
                piArray[i] = Math.log(weightSum + smoothing) - piLog;
            }

            NaiveBayesModelData modelData = new NaiveBayesModelData(
                    featureColNames,
                    theta,
                    piArray,
                    labels
            );
            out.collect(modelData);
        }
    }
}
