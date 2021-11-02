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
import org.apache.flink.ml.dataproc.stringindexer.MultiStringIndexer;
import org.apache.flink.ml.dataproc.stringindexer.MultiStringIndexerModel;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.WithParams;
import org.apache.flink.ml.param.shared.colname.HasCategoricalCols;
import org.apache.flink.ml.util.TableUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.functions.windowing.AllWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.expressions.Expression;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import static org.apache.flink.table.api.Expressions.$;

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
        Preconditions.checkNotNull(inputs, "input table list should not be null");
        Preconditions.checkArgument(inputs.length == 1, "input table list should contain only one argument");

        String[] featureColNames = getFeatureCols();
        Preconditions.checkArgument(
                new HashSet<>(Arrays.asList(featureColNames)).size() == featureColNames.length,
                "feature columns should not duplicate");
        String labelColName = getLabelCol();
        Preconditions.checkNotNull(labelColName, "label column should be set");
        String weightColName = getWeightCol() == null ? "" : getWeightCol();
        int featureSize = featureColNames.length;
        String[] originalCategoricalCols = getUserDefinedParamMap().containsKey(HasCategoricalCols.CATEGORICAL_COLS) ?
                getCategoricalCols() : new String[0];
        Preconditions.checkArgument(
                new HashSet<>(Arrays.asList(originalCategoricalCols)).size() == originalCategoricalCols.length,
                "categorical columns should not duplicate");
        Preconditions.checkArgument(
                new HashSet<>(Arrays.asList(featureColNames)).containsAll(new HashSet<>(Arrays.asList(originalCategoricalCols))),
                "feature columns should contain all categorical columns");
        double smoothing = getSmoothing();

        // get data types of feature columns.
        DataType[] featureTypes = new DataType[featureSize];
        List<DataType> inputDataTypes = inputs[0].getResolvedSchema().getColumnDataTypes();
        List<String> inputColNames = inputs[0].getResolvedSchema().getColumnNames();
        for (int i = 0; i < featureSize; i++) {
            featureTypes[i] = inputDataTypes.get(inputColNames.indexOf(featureColNames[i]));
        }

        boolean[] isCate = generateCategoricalCols(
                originalCategoricalCols,
                featureColNames,
                featureTypes,
                labelColName,
                this);

        // convert string-typed features to indexed numbers.
        List<String> stringColNames = new ArrayList<>();
        for (int i = 0; i < featureSize; i++) {
            if (featureTypes[i].equals(DataTypes.STRING())) {
                stringColNames.add(featureColNames[i]);
            }
        }
        MultiStringIndexerModel multiStringIndexerModel = null;
        if (!stringColNames.isEmpty()) {
            Expression[] expressions = inputs[0]
                    .getResolvedSchema()
                    .getColumnNames()
                    .stream()
                    .map((Function<String, Expression>) s -> $(s).as("raw-" + s))
                    .toArray(Expression[]::new);
            inputs[0] = inputs[0].select(expressions);

            MultiStringIndexer multiStringIndexer = new MultiStringIndexer()
                    .setSelectedCols(stringColNames.stream().map(s -> "raw-" + s).toArray(String[]::new))
                    .setOutputCols(stringColNames.stream().map(s -> "indexed-" + s).toArray(String[]::new))
                    .setReservedCols(inputs[0].getResolvedSchema().getColumnNames().toArray(new String[0]));
            multiStringIndexerModel = multiStringIndexer.fit(inputs);
            inputs = multiStringIndexerModel.transform(inputs);
            if (!weightColName.equals("")) weightColName = "raw-" + weightColName;
            labelColName = "raw-" + labelColName;
            for (int i = 0; i < featureColNames.length; i++) {
                if (stringColNames.contains(featureColNames[i])) {
                    featureColNames[i] = "indexed-" + featureColNames[i];
                } else {
                    featureColNames[i] = "raw-" + featureColNames[i];
                }
            }
        }

        StreamTableEnvironment tEnv = (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);

        DataStream<NaiveBayesModelData> naiveBayesModel = input
                .flatMap(new FlattenFunction(
                        weightColName,
                        featureColNames,
                        labelColName,
                        isCate
                ))
                .keyBy(new SumWeightKeySelector())
                .window(EndOfStreamWindows.get())
                .reduce(new SumWeightFunction())
                .keyBy(new ValueMapKeySelector())
                .window(EndOfStreamWindows.get())
                .aggregate(new ValueMapFunction())
                .keyBy(new MapArrayKeySelector())
                .window(EndOfStreamWindows.get())
                .aggregate(new MapArrayFunction(featureSize))
                .windowAll(EndOfStreamWindows.get())
                .apply(new GenerateModelFunction(
                        smoothing,
                        featureColNames,
                        isCate));

        NaiveBayesModel model = new NaiveBayesModel(multiStringIndexerModel)
                .setPredictionCol(getPredictionCol())
                .setReservedCols(getReservedCols());
        model.setModelData(
                tEnv.fromDataStream(
                        naiveBayesModel.flatMap(
                                new NaiveBayesUtils.Serializer()))
        );
        return model;
    }

    @Override
    public void save(String path) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Param<?>, Object> getUserDefinedParamMap() {
        return paramMap;
    }

    private static boolean[] generateCategoricalCols(String[] originCategoricalColNames,
                                                     String[] inputColNames, DataType[] inputTypes,
                                                     String labelColName, WithParams<NaiveBayes> params) {
        Set<DataType> trueSet = new HashSet<>(Arrays.asList(DataTypes.STRING(), DataTypes.BOOLEAN()));
        Set<DataType> falseSet = new HashSet<>(Arrays.asList(DataTypes.DOUBLE(), DataTypes.FLOAT()));
        // ori impl also contains Types.LONG, but DataTypes does not have this type
        Set<DataType> doubtSet = new HashSet<>(Arrays.asList(DataTypes.BIGINT(), DataTypes.INT()));

        List<String> categoricalCols = new ArrayList<>();
        int length = inputColNames.length;
        boolean[] isCate = new boolean[length];
        for (int i = 0; i < length; i++) {
            String colName = inputColNames[i];
            if (colName.equals(labelColName)) {
                continue;
            }
            DataType type = inputTypes[i];

            if (trueSet.contains(type)) {
                categoricalCols.add(colName);
                isCate[i] = true;
            } else if (falseSet.contains(type)) {
                if (TableUtils.findColIndex(originCategoricalColNames, colName) != -1) {
                    throw new RuntimeException("column \"" + colName + "\"'s type is " + type +
                            ", which is not categorical!");
                }
            } else if (doubtSet.contains(type)) {
                if (TableUtils.findColIndex(originCategoricalColNames, colName) != -1) {
                    categoricalCols.add(colName);
                    isCate[i] = true;
                }
            } else {
                throw new RuntimeException("don't support the type " + type);
            }
        }
        params.set(HasCategoricalCols.CATEGORICAL_COLS, categoricalCols.toArray(new String[0]));
        return isCate;
    }

    // converts each column into tuples of label, feature column index, feature value, weight
    private static class FlattenFunction implements FlatMapFunction<Row, Tuple4<Object, Integer, Number, Double>> {
        private final String weightColName;
        private final String[] featureColNames;
        private final String labelColName;
        private final int featureSize;
        boolean[] isCate;

        private FlattenFunction(String weightColName, String[] featureColNames, String labelColName, boolean[] isCate) {
            this.labelColName = labelColName;
            this.weightColName = weightColName;
            this.featureColNames = featureColNames;
            this.featureSize = featureColNames.length;
            this.isCate = isCate;
        }

        @Override
        public void flatMap(Row row, Collector<Tuple4<Object, Integer, Number, Double>> collector) {
            Double weight;
            if (weightColName == null || weightColName.equals("") || row.getField(weightColName) == null) {
                weight = 1.0;
            } else {
                weight = (Double) row.getField(weightColName);
            }
            Object label = row.getField(labelColName);
            if (label == null) {
                return;
            }

            for (int i = 0; i < featureSize; i++) {
                Number feature = (Number) row.getField(featureColNames[i]);
                if (feature == null) {
                    continue;
                }
                if (isCate[i]) {
                    collector.collect(new Tuple4<>(label, i, feature, weight));
                } else {
                    collector.collect(new Tuple4<>(label, i, 0, weight * feature.doubleValue()));
                    collector.collect(new Tuple4<>(label, i, 1, weight * Math.pow(feature.doubleValue(), 2)));
                }
            }
        }
    }

    private static class SumWeightKeySelector implements KeySelector<
            Tuple4<Object, Integer, Number, Double>,
            Tuple3<Object, Integer, Number>> {
        @Override
        public Tuple3<Object, Integer, Number> getKey(Tuple4<Object, Integer, Number, Double> value) {
            return new Tuple3<>(value.f0, value.f1, value.f2);
        }
    }

    // sums weight value for records with the same label, feature column index, feature value, weightSum
    private static class SumWeightFunction implements ReduceFunction<Tuple4<Object, Integer, Number, Double>> {
        @Override
        public Tuple4<Object, Integer, Number, Double> reduce(
                Tuple4<Object, Integer, Number, Double> t0,
                Tuple4<Object, Integer, Number, Double> t1) {
            t0.f3 += t1.f3;
            return t0;
        }
    }

    private static class ValueMapKeySelector implements KeySelector<
            Tuple4<Object, Integer, Number, Double>,
            Tuple2<Object, Integer>
            > {
        @Override
        public Tuple2<Object, Integer> getKey(Tuple4<Object, Integer, Number, Double> value) {
            return new Tuple2<>(value.f0, value.f1);
        }
    }

    // aggregates feature value and weight into map from records with the same label and feature column index
    private static class ValueMapFunction implements AggregateFunction<
            Tuple4<Object, Integer, Number, Double>,
            Tuple4<Object, Integer, Map<Integer, Double>, Double>,
            Tuple4<Object, Integer, Map<Integer, Double>, Double>> {

        @Override
        public Tuple4<Object, Integer, Map<Integer, Double>, Double> createAccumulator() {
            return new Tuple4<>(new Object(), -1, new HashMap<>(), 0.);
        }

        @Override
        public Tuple4<Object, Integer, Map<Integer, Double>, Double> add(
                Tuple4<Object, Integer, Number, Double> value,
                Tuple4<Object, Integer, Map<Integer, Double>, Double> acc) {
            acc.f0 = value.f0;
            acc.f1 = value.f1;
            acc.f2.put((Integer) value.f2, value.f3);
            acc.f3 += value.f3;
            return acc;
        }

        @Override
        public Tuple4<Object, Integer, Map<Integer, Double>, Double> getResult(
                Tuple4<Object, Integer, Map<Integer, Double>, Double> acc) {
            return acc;
        }

        @Override
        public Tuple4<Object, Integer, Map<Integer, Double>, Double> merge(
                Tuple4<Object, Integer, Map<Integer, Double>, Double> acc0,
                Tuple4<Object, Integer, Map<Integer, Double>, Double> acc1) {
            if (acc0.f0.equals(new Object())) {
                acc0.f0 = acc1.f0;
            }
            if (acc0.f1 == -1) {
                acc0.f1 = acc1.f1;
            }
            // feature value has been deduplicated in previous stages.
            for (Map.Entry<Integer, Double> entry: acc1.f2.entrySet()) {
                acc0.f2.put(entry.getKey(), entry.getValue());
            }
            acc0.f3 += acc1.f3;

            return acc0;
        }
    }

    private static class MapArrayKeySelector implements KeySelector<
            Tuple4<Object, Integer, Map<Integer, Double>, Double>,
            Object
            > {

        @Override
        public Object getKey(Tuple4<Object, Integer, Map<Integer, Double>, Double> value) {
            return value.f0;
        }
    }

    // aggregate maps under the same label into arrays. array len = featureSize
    private static class MapArrayFunction implements AggregateFunction<
            Tuple4<Object, Integer, Map<Integer, Double>, Double>,
            Tuple3<Object, Double[], Map <Integer, Double>[]>,
            Tuple3<Object, Double[], Map <Integer, Double>[]>> {
        private final int featureSize;

        private MapArrayFunction(int featureSize) {
            this.featureSize = featureSize;
        }

        @Override
        public Tuple3<Object, Double[], Map<Integer, Double>[]> createAccumulator() {
            Double[] weightSum = new Double[featureSize];
            Arrays.fill(weightSum, 0.);
            return new Tuple3<>(new Object(), weightSum, new HashMap[featureSize]);
        }

        @Override
        public Tuple3<Object, Double[], Map<Integer, Double>[]> add(
                Tuple4<Object, Integer, Map<Integer, Double>, Double> value,
                Tuple3<Object, Double[], Map<Integer, Double>[]> acc) {
            acc.f0 = value.f0;
            acc.f1[value.f1] = value.f3;
            acc.f2[value.f1] = value.f2;

            return acc;
        }

        @Override
        public Tuple3<Object, Double[], Map<Integer, Double>[]> getResult(Tuple3<Object, Double[], Map<Integer, Double>[]> acc) {
            return acc;
        }

        @Override
        public Tuple3<Object, Double[], Map<Integer, Double>[]> merge(
                Tuple3<Object, Double[], Map<Integer, Double>[]> acc0,
                Tuple3<Object, Double[], Map<Integer, Double>[]> acc1) {
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

    // generate Naive Bayes model.
    private static class GenerateModelFunction implements AllWindowFunction<
            Tuple3<Object, Double[], Map<Integer, Double>[]>,
            NaiveBayesModelData,
            TimeWindow
            > {
        private final int featureSize;
        private final double smoothing;
        private final String[] featureColNames;
        private final boolean[] isCate;

        GenerateModelFunction(double smoothing, String[] featureColNames, boolean[] isCate) {
            this.smoothing = smoothing;
            this.featureColNames = featureColNames;
            this.isCate = isCate;
            this.featureSize = featureColNames.length;
        }

        @Override
        public void apply(TimeWindow timeWindow,
                          Iterable<Tuple3<Object, Double[], Map<Integer, Double>[]>> values,
                          Collector<NaiveBayesModelData> out) {
            double[] numDocs = new double[featureSize];
            ArrayList <Tuple3 <Object, Double[], Map <Integer, Double>[]>> modelArray = new ArrayList <>();
            HashSet <Integer>[] categoryNumbers = new HashSet[featureSize];
            for (int i = 0; i < featureSize; i++) {
                categoryNumbers[i] = new HashSet <>();
            }
            for (Tuple3 <Object, Double[], Map <Integer, Double>[]> tup : values) {
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

            Number[][][] theta = new Number[numLabels][featureSize][];
            double[] piArray = new double[numLabels];
            double[] pi = new double[numLabels];
            Object[] labels = new Object[numLabels];

            //consider smoothing.
            for (int i = 0; i < numLabels; i++) {
                Map <Integer, Double>[] param = modelArray.get(i).f2;
                for (int j = 0; j < featureSize; j++) {
                    int size = categoryNumber[j];
                    Number[] squareData = new Number[size];
                    if (isCate[j]) {
                        double thetaLog = Math.log(modelArray.get(i).f1[j] + smoothing * categoryNumber[j]);
                        for (int k = 0; k < size; k++) {
                            double value = 0;
                            if (param[j].containsKey(k)) {
                                value = param[j].get(k);
                            }
                            squareData[k] = Math.log(value + smoothing) - thetaLog;
                        }
                    } else {
                        for (int k = 0; k < size; k++) {
                            squareData[k] = param[j].get(k);
                        }
                    }
                    theta[i][j] = squareData;
                }

                labels[i] = modelArray.get(i).f0;
                double weightSum = 0;
                for (Double weight : modelArray.get(i).f1) {
                    weightSum += weight;
                }
                pi[i] = weightSum;
                piArray[i] = Math.log(weightSum + smoothing) - piLog;
            }
            NaiveBayesModelData modelData = new NaiveBayesModelData();
            modelData.featureNames = featureColNames;
            modelData.isCate = isCate;
            modelData.label = labels;
            modelData.piArray = piArray;
            modelData.labelWeights = pi;
            modelData.theta = theta;
            modelData.generateWeightAndNumbers(modelArray);
            out.collect(modelData);
        }
    }
}
