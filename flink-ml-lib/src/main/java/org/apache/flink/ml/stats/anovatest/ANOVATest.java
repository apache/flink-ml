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

package org.apache.flink.ml.stats.anovatest;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.param.HasFlatten;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.math3.distribution.FDistribution;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * An AlgoOperator which implements the ANOVA test algorithm.
 *
 * <p>See <a href="https://en.wikipedia.org/wiki/Analysis_of_variance">Wikipedia</a> for more
 * information on ANOVA test.
 *
 * <p>The input of this algorithm is a table containing a labelColumn of numerical type and a
 * featuresColumn of vector type. Each index in the input vector represents a feature to be tested.
 * By default, the output of this algorithm is a table containing a single row with the following
 * columns, each of which has one value per feature.
 *
 * <ul>
 *   <li>"pValues": vector
 *   <li>"degreesOfFreedom": int array
 *   <li>"fValues": vector
 * </ul>
 *
 * <p>The output of this algorithm can be flattened to multiple rows by setting {@link
 * HasFlatten#FLATTEN} to true, which would contain the following columns:
 *
 * <ul>
 *   <li>"featureIndex": int
 *   <li>"pValue": double
 *   <li>"degreeOfFreedom": int
 *   <li>"fValues": double
 * </ul>
 */
public class ANOVATest implements AlgoOperator<ANOVATest>, ANOVATestParams<ANOVATest> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public ANOVATest() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        final String featuresCol = getFeaturesCol();
        final String labelCol = getLabelCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<Tuple2<Vector, Double>> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, Tuple2<Vector, Double>>)
                                        row -> {
                                            Number number = (Number) row.getField(labelCol);
                                            Preconditions.checkNotNull(
                                                    number, "Input data must contain label value.");
                                            return new Tuple2<>(
                                                    ((Vector) row.getField(featuresCol)),
                                                    number.doubleValue());
                                        },
                                Types.TUPLE(VectorTypeInfo.INSTANCE, Types.DOUBLE));
        DataStream<List<Row>> streamWithANOVA =
                DataStreamUtils.aggregate(
                        inputData,
                        new ANOVAAggregator(),
                        Types.OBJECT_ARRAY(
                                Types.TUPLE(
                                        Types.DOUBLE,
                                        Types.DOUBLE,
                                        Types.MAP(
                                                Types.DOUBLE,
                                                Types.TUPLE(Types.DOUBLE, Types.LONG)))),
                        Types.LIST(Types.ROW(Types.INT, Types.DOUBLE, Types.LONG, Types.DOUBLE)));
        return new Table[] {convertToTable(tEnv, streamWithANOVA, getFlatten())};
    }

    /** Computes the p-value, fValues and the number of degrees of freedom of input features. */
    @SuppressWarnings("unchecked")
    private static class ANOVAAggregator
            implements AggregateFunction<
                    Tuple2<Vector, Double>,
                    Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[],
                    List<Row>> {
        @Override
        public Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[] createAccumulator() {
            return new Tuple3[0];
        }

        @Override
        public Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[] add(
                Tuple2<Vector, Double> featuresAndLabel,
                Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[] acc) {
            Vector features = featuresAndLabel.f0;
            double label = featuresAndLabel.f1;
            int numOfFeatures = features.size();
            if (acc.length == 0) {
                acc = new Tuple3[features.size()];
                for (int i = 0; i < numOfFeatures; i++) {
                    acc[i] = Tuple3.of(0.0, 0.0, new HashMap<>());
                }
            }
            for (int i = 0; i < numOfFeatures; i++) {
                double featureValue = features.get(i);
                acc[i].f0 += featureValue;
                acc[i].f1 += featureValue * featureValue;

                if (acc[i].f2.containsKey(label)) {
                    acc[i].f2.get(label).f0 += featureValue;
                    acc[i].f2.get(label).f1 += 1L;
                } else {
                    acc[i].f2.put(label, Tuple2.of(featureValue, 1L));
                }
            }
            return acc;
        }

        @Override
        public List<Row> getResult(
                Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[] acc) {
            List<Row> results = new ArrayList<>();
            for (int i = 0; i < acc.length; i++) {
                Tuple3<Double, Long, Double> resultOfANOVA =
                        computeANOVA(acc[i].f0, acc[i].f1, acc[i].f2);
                results.add(Row.of(i, resultOfANOVA.f0, resultOfANOVA.f1, resultOfANOVA.f2));
            }
            return results;
        }

        @Override
        public Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[] merge(
                Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[] acc1,
                Tuple3<Double, Double, HashMap<Double, Tuple2<Double, Long>>>[] acc2) {
            if (acc1.length == 0) {
                return acc2;
            }
            if (acc2.length == 0) {
                return acc1;
            }
            IntStream.range(0, acc1.length)
                    .forEach(
                            i -> {
                                acc2[i].f0 += acc1[i].f0;
                                acc2[i].f1 += acc1[i].f1;
                                acc1[i].f2.forEach(
                                        (k, v) -> {
                                            if (acc2[i].f2.containsKey(k)) {
                                                acc2[i].f2.get(k).f0 += v.f0;
                                                acc2[i].f2.get(k).f1 += v.f1;
                                            } else {
                                                acc2[i].f2.put(k, v);
                                            }
                                        });
                            });
            return acc2;
        }

        private Tuple3<Double, Long, Double> computeANOVA(
                double sum, double sumOfSq, HashMap<Double, Tuple2<Double, Long>> summary) {
            long numOfClasses = summary.size();

            long numOfSamples = summary.values().stream().mapToLong(t -> t.f1).sum();

            double sqSum = sum * sum;

            double ssTot = sumOfSq - sqSum / numOfSamples;

            double totalSqSum = 0;
            for (Tuple2<Double, Long> t : summary.values()) {
                totalSqSum += t.f0 * t.f0 / t.f1;
            }

            double sumOfSqBetween = totalSqSum - (sqSum / numOfSamples);

            double sumOfSqWithin = ssTot - sumOfSqBetween;

            long degreeOfFreedomBetween = numOfClasses - 1;
            Preconditions.checkArgument(
                    degreeOfFreedomBetween > 0, "Num of classes should be positive.");

            long degreeOfFreedomWithin = numOfSamples - numOfClasses;
            Preconditions.checkArgument(
                    degreeOfFreedomWithin > 0,
                    "Num of samples should be greater than num of classes.");

            double meanSqBetween = sumOfSqBetween / degreeOfFreedomBetween;

            double meanSqWithin = sumOfSqWithin / degreeOfFreedomWithin;

            double fValue = meanSqBetween / meanSqWithin;

            FDistribution fd = new FDistribution(degreeOfFreedomBetween, degreeOfFreedomWithin);
            double pValue = 1 - fd.cumulativeProbability(fValue);

            long degreeOfFreedom = degreeOfFreedomBetween + degreeOfFreedomWithin;

            return Tuple3.of(pValue, degreeOfFreedom, fValue);
        }
    }

    private Table convertToTable(
            StreamTableEnvironment tEnv, DataStream<List<Row>> datastream, boolean flatten) {
        if (flatten) {
            DataStream<Row> output =
                    datastream
                            .flatMap(
                                    (FlatMapFunction<List<Row>, Row>)
                                            (list, collector) -> list.forEach(collector::collect))
                            .setParallelism(1)
                            .returns(Types.ROW(Types.INT, Types.DOUBLE, Types.LONG, Types.DOUBLE));
            return tEnv.fromDataStream(output)
                    .as("featureIndex", "pValue", "degreeOfFreedom", "fValue");
        } else {
            DataStream<Tuple3<DenseVector, long[], DenseVector>> output =
                    datastream.map(
                            new MapFunction<List<Row>, Tuple3<DenseVector, long[], DenseVector>>() {
                                @Override
                                public Tuple3<DenseVector, long[], DenseVector> map(
                                        List<Row> rows) {
                                    int numOfFeatures = rows.size();
                                    DenseVector pValues = new DenseVector(numOfFeatures);
                                    DenseVector fValues = new DenseVector(numOfFeatures);
                                    long[] degrees = new long[numOfFeatures];

                                    for (int i = 0; i < numOfFeatures; i++) {
                                        Row row = rows.get(i);
                                        pValues.set(i, (double) row.getField(1));
                                        degrees[i] = (long) row.getField(2);
                                        fValues.set(i, (double) row.getField(3));
                                    }
                                    return Tuple3.of(pValues, degrees, fValues);
                                }
                            });
            return tEnv.fromDataStream(output).as("pValues", "degreesOfFreedom", "fValues");
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static ANOVATest load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
