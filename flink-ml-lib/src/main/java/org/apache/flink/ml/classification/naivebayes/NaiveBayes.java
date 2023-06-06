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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * An Estimator which implements the naive bayes classification algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/Naive_Bayes_classifier.
 */
public class NaiveBayes
        implements Estimator<NaiveBayes, NaiveBayesModel>, NaiveBayesParams<NaiveBayes> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public NaiveBayes() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public NaiveBayesModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        final String featuresCol = getFeaturesCol();
        final String labelCol = getLabelCol();
        final double smoothing = getSmoothing();

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Tuple2<IntDoubleVector, Double>> input =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, Tuple2<IntDoubleVector, Double>>)
                                        row -> {
                                            Number number = (Number) row.getField(labelCol);
                                            Preconditions.checkNotNull(
                                                    number,
                                                    "Input data should contain label value.");
                                            Preconditions.checkArgument(
                                                    number.intValue() == number.doubleValue(),
                                                    "Label value should be indexed number.");
                                            return new Tuple2<>(
                                                    (IntDoubleVector) row.getField(featuresCol),
                                                    number.doubleValue());
                                        },
                                Types.TUPLE(VectorTypeInfo.INSTANCE, Types.DOUBLE));

        DataStream<Tuple3<Double, Integer, Double>> feature =
                input.flatMap(new ExtractFeatureFunction());

        DataStream<Tuple4<Double, Integer, Map<Double, Double>, Integer>> featureWeight =
                DataStreamUtils.mapPartition(
                        feature.keyBy(value -> new Tuple2<>(value.f0, value.f1).hashCode()),
                        new GenerateFeatureWeightMapFunction(),
                        Types.TUPLE(
                                Types.DOUBLE,
                                Types.INT,
                                Types.MAP(Types.DOUBLE, Types.DOUBLE),
                                Types.INT));

        DataStream<Tuple3<Double, Integer, Map<Double, Double>[]>> aggregatedArrays =
                DataStreamUtils.mapPartition(
                        featureWeight.keyBy(value -> value.f0),
                        new AggregateIntoArrayFunction(),
                        Types.TUPLE(
                                Types.DOUBLE,
                                Types.INT,
                                Types.OBJECT_ARRAY(Types.MAP(Types.DOUBLE, Types.DOUBLE))));

        DataStream<NaiveBayesModelData> modelData =
                DataStreamUtils.mapPartition(
                        aggregatedArrays,
                        new GenerateModelFunction(smoothing),
                        NaiveBayesModelData.TYPE_INFO);
        modelData.getTransformation().setParallelism(1);

        Schema schema =
                Schema.newBuilder()
                        .column(
                                "theta",
                                DataTypes.ARRAY(
                                        DataTypes.ARRAY(
                                                DataTypes.MAP(
                                                        DataTypes.DOUBLE(), DataTypes.DOUBLE()))))
                        .column("piArray", DataTypes.of(DenseIntDoubleVectorTypeInfo.INSTANCE))
                        .column("labels", DataTypes.of(DenseIntDoubleVectorTypeInfo.INSTANCE))
                        .build();

        NaiveBayesModel model =
                new NaiveBayesModel().setModelData(tEnv.fromDataStream(modelData, schema));
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static NaiveBayes load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Function to extract feature values from input rows.
     *
     * <p>Output records are tuples with the following fields in order:
     *
     * <ul>
     *   <li>label value
     *   <li>feature column index
     *   <li>feature value
     * </ul>
     */
    private static class ExtractFeatureFunction
            implements FlatMapFunction<
                    Tuple2<IntDoubleVector, Double>, Tuple3<Double, Integer, Double>> {
        @Override
        public void flatMap(
                Tuple2<IntDoubleVector, Double> value,
                Collector<Tuple3<Double, Integer, Double>> collector) {
            Preconditions.checkNotNull(value.f1);
            for (int i = 0; i < value.f0.size(); i++) {
                collector.collect(new Tuple3<>(value.f1, i, value.f0.get(i)));
            }
        }
    }

    /**
     * Function that aggregates entries of feature value and weight into maps.
     *
     * <p>Input records should have the same label value and feature column index.
     *
     * <p>Input records are tuples with the following fields in order:
     *
     * <ul>
     *   <li>label value
     *   <li>feature column index
     *   <li>feature value
     * </ul>
     *
     * <p>Output records are tuples with the following fields in order:
     *
     * <ul>
     *   <li>label value
     *   <li>feature column index
     *   <li>map of (feature value, weight)
     *   <li>number of records
     * </ul>
     */
    private static class GenerateFeatureWeightMapFunction
            implements MapPartitionFunction<
                    Tuple3<Double, Integer, Double>,
                    Tuple4<Double, Integer, Map<Double, Double>, Integer>> {

        @Override
        public void mapPartition(
                Iterable<Tuple3<Double, Integer, Double>> iterable,
                Collector<Tuple4<Double, Integer, Map<Double, Double>, Integer>> collector) {
            List<Tuple3<Double, Integer, Double>> list = new ArrayList<>();
            iterable.iterator().forEachRemaining(list::add);

            Map<Tuple2<Double, Integer>, Map<Double, Double>> accMap = new HashMap<>();
            Map<Tuple2<Double, Integer>, Integer> numMap = new HashMap<>();
            for (Tuple3<Double, Integer, Double> value : list) {
                Tuple2<Double, Integer> key = new Tuple2<>(value.f0, value.f1);
                Map<Double, Double> acc = accMap.computeIfAbsent(key, x -> new HashMap<>());
                acc.put(value.f2, acc.getOrDefault(value.f2, 0.) + 1.0);
                numMap.put(key, numMap.getOrDefault(key, 0) + 1);
            }
            for (Map.Entry<Tuple2<Double, Integer>, Map<Double, Double>> entry :
                    accMap.entrySet()) {
                collector.collect(
                        new Tuple4<>(
                                entry.getKey().f0,
                                entry.getKey().f1,
                                entry.getValue(),
                                numMap.get(entry.getKey())));
            }
        }
    }

    /**
     * Function that aggregates maps under the same label into arrays.
     *
     * <p>Length of the generated array equals to the number of feature columns.
     *
     * <p>Input records are tuples with the following fields in order:
     *
     * <ul>
     *   <li>label value
     *   <li>feature column index
     *   <li>map of (feature value, weight)
     *   <li>number of records
     * </ul>
     *
     * <p>Output records are tuples with the following fields in order:
     *
     * <ul>
     *   <li>label value
     *   <li>number of records
     *   <li>array of featureValue-weight maps of each feature
     * </ul>
     */
    private static class AggregateIntoArrayFunction
            implements MapPartitionFunction<
                    Tuple4<Double, Integer, Map<Double, Double>, Integer>,
                    Tuple3<Double, Integer, Map<Double, Double>[]>> {

        @Override
        public void mapPartition(
                Iterable<Tuple4<Double, Integer, Map<Double, Double>, Integer>> iterable,
                Collector<Tuple3<Double, Integer, Map<Double, Double>[]>> collector) {
            Map<Double, List<Tuple4<Double, Integer, Map<Double, Double>, Integer>>> map =
                    new HashMap<>();
            for (Tuple4<Double, Integer, Map<Double, Double>, Integer> value : iterable) {
                map.computeIfAbsent(value.f0, x -> new ArrayList<>()).add(value);
            }

            for (List<Tuple4<Double, Integer, Map<Double, Double>, Integer>> list : map.values()) {
                final int featureSize =
                        list.stream().map(x -> x.f1).max(Integer::compareTo).orElse(-1) + 1;

                int minDocNum =
                        list.stream()
                                .map(x -> x.f3)
                                .min(Integer::compareTo)
                                .orElse(Integer.MAX_VALUE);
                int maxDocNum =
                        list.stream()
                                .map(x -> x.f3)
                                .max(Integer::compareTo)
                                .orElse(Integer.MIN_VALUE);
                Preconditions.checkArgument(
                        minDocNum == maxDocNum, "Feature vectors should be of equal length.");

                Map<Double, Integer> numMap = new HashMap<>();
                Map<Double, Map<Double, Double>[]> featureWeightMap = new HashMap<>();
                for (Tuple4<Double, Integer, Map<Double, Double>, Integer> value : list) {
                    Map<Double, Double>[] featureWeight =
                            featureWeightMap.computeIfAbsent(
                                    value.f0, x -> new HashMap[featureSize]);
                    numMap.put(value.f0, value.f3);
                    featureWeight[value.f1] = value.f2;
                }

                for (double key : featureWeightMap.keySet()) {
                    collector.collect(
                            new Tuple3<>(key, numMap.get(key), featureWeightMap.get(key)));
                }
            }
        }
    }

    /** Function to generate Naive Bayes model data. */
    private static class GenerateModelFunction
            implements MapPartitionFunction<
                    Tuple3<Double, Integer, Map<Double, Double>[]>, NaiveBayesModelData> {
        private final double smoothing;

        private GenerateModelFunction(double smoothing) {
            this.smoothing = smoothing;
        }

        @Override
        public void mapPartition(
                Iterable<Tuple3<Double, Integer, Map<Double, Double>[]>> iterable,
                Collector<NaiveBayesModelData> collector) {
            ArrayList<Tuple3<Double, Integer, Map<Double, Double>[]>> list = new ArrayList<>();
            iterable.iterator().forEachRemaining(list::add);
            final int featureSize = list.get(0).f2.length;
            for (Tuple3<Double, Integer, Map<Double, Double>[]> tup : list) {
                Preconditions.checkArgument(
                        featureSize == tup.f2.length, "Feature vectors should be of equal length.");
            }

            double[] numDocs = new double[featureSize];
            HashSet<Double>[] categoryNumbers = new HashSet[featureSize];
            for (int i = 0; i < featureSize; i++) {
                categoryNumbers[i] = new HashSet<>();
            }
            for (Tuple3<Double, Integer, Map<Double, Double>[]> tup : list) {
                for (int i = 0; i < featureSize; i++) {
                    numDocs[i] += tup.f1;
                    categoryNumbers[i].addAll(tup.f2[i].keySet());
                }
            }

            int[] categoryNumber = new int[featureSize];
            double piLog = 0;
            int numLabels = list.size();
            for (int i = 0; i < featureSize; i++) {
                categoryNumber[i] = categoryNumbers[i].size();
                piLog += numDocs[i];
            }
            piLog = Math.log(piLog + numLabels * smoothing);

            Map<Double, Double>[][] theta = new HashMap[numLabels][featureSize];
            double[] piArray = new double[numLabels];
            double[] labels = new double[numLabels];

            // Consider smoothing.
            for (int i = 0; i < numLabels; i++) {
                Map<Double, Double>[] param = list.get(i).f2;
                for (int j = 0; j < featureSize; j++) {
                    Map<Double, Double> squareData = new HashMap<>();
                    double thetaLog =
                            Math.log(list.get(i).f1 * 1.0 + smoothing * categoryNumber[j]);
                    for (Double cate : categoryNumbers[j]) {
                        double value = param[j].getOrDefault(cate, 0.0);
                        squareData.put(cate, Math.log(value + smoothing) - thetaLog);
                    }
                    theta[i][j] = squareData;
                }

                labels[i] = list.get(i).f0;
                double weightSum = list.get(i).f1 * featureSize;
                piArray[i] = Math.log(weightSum + smoothing) - piLog;
            }

            NaiveBayesModelData modelData =
                    new NaiveBayesModelData(theta, Vectors.dense(piArray), Vectors.dense(labels));
            collector.collect(modelData);
        }
    }
}
