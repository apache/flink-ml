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

package org.apache.flink.ml.stats.fvaluetest;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.tuple.Tuple5;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.param.HasFlatten;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.VectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.math3.distribution.FDistribution;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An AlgoOperator which implements the F-value test algorithm.
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
public class FValueTest implements AlgoOperator<FValueTest>, FValueTestParams<FValueTest> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public FValueTest() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @SuppressWarnings("unchecked, rawtypes")
    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        final String featuresCol = getFeaturesCol();
        final String labelCol = getLabelCol();
        final String broadcastSummaryKey = "broadcastSummaryKey";
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
                                        })
                        .returns(Types.TUPLE(VectorTypeInfo.INSTANCE, Types.DOUBLE));

        DataStream<Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector>>
                summaries = DataStreamUtils.aggregate(inputData, new SummaryAggregator());

        DataStream<DenseIntDoubleVector> covarianceInEachPartition =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputData),
                        Collections.singletonMap(broadcastSummaryKey, summaries),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return DataStreamUtils.mapPartition(
                                    input, new CalCovarianceOperator(broadcastSummaryKey));
                        });

        DataStream<DenseIntDoubleVector> reducedCovariance =
                DataStreamUtils.reduce(
                        covarianceInEachPartition,
                        (ReduceFunction<DenseIntDoubleVector>)
                                (sums1, sums2) -> {
                                    BLAS.axpy(1.0, sums1, sums2);
                                    return sums2;
                                });

        DataStream result =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(reducedCovariance),
                        Collections.singletonMap(broadcastSummaryKey, summaries),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return DataStreamUtils.mapPartition(
                                    input, new CalFValueOperator(broadcastSummaryKey));
                        });

        return new Table[] {convertToTable(tEnv, result, getFlatten())};
    }

    private Table convertToTable(
            StreamTableEnvironment tEnv,
            DataStream<Tuple4<Integer, Double, Long, Double>> dataStream,
            boolean flatten) {
        if (flatten) {
            return tEnv.fromDataStream(dataStream)
                    .as("featureIndex", "pValue", "degreeOfFreedom", "fValue");
        } else {
            DataStream<Tuple3<DenseIntDoubleVector, long[], DenseIntDoubleVector>> output =
                    DataStreamUtils.mapPartition(
                            dataStream,
                            new MapPartitionFunction<
                                    Tuple4<Integer, Double, Long, Double>,
                                    Tuple3<DenseIntDoubleVector, long[], DenseIntDoubleVector>>() {
                                @Override
                                public void mapPartition(
                                        Iterable<Tuple4<Integer, Double, Long, Double>> iterable,
                                        Collector<
                                                        Tuple3<
                                                                DenseIntDoubleVector,
                                                                long[],
                                                                DenseIntDoubleVector>>
                                                collector) {
                                    List<Tuple4<Integer, Double, Long, Double>> rows =
                                            IteratorUtils.toList(iterable.iterator());
                                    int numOfFeatures = rows.size();

                                    DenseIntDoubleVector pValues = Vectors.dense(numOfFeatures);
                                    long[] degrees = new long[numOfFeatures];
                                    DenseIntDoubleVector fValues = Vectors.dense(numOfFeatures);

                                    for (int i = 0; i < numOfFeatures; i++) {
                                        Tuple4<Integer, Double, Long, Double> tuple = rows.get(i);
                                        pValues.set(i, tuple.f1.doubleValue());
                                        degrees[i] = tuple.f2;
                                        fValues.set(i, tuple.f3.doubleValue());
                                    }
                                    collector.collect(Tuple3.of(pValues, degrees, fValues));
                                }
                            });
            return tEnv.fromDataStream(output).as("pValues", "degreesOfFreedom", "fValues");
        }
    }

    /** Computes the covariance of each feature on each partition. */
    private static class CalCovarianceOperator
            extends RichMapPartitionFunction<Tuple2<Vector, Double>, DenseIntDoubleVector> {

        private final String broadcastKey;

        private CalCovarianceOperator(String broadcastKey) {
            this.broadcastKey = broadcastKey;
        }

        @Override
        public void mapPartition(
                Iterable<Tuple2<Vector, Double>> iterable,
                Collector<DenseIntDoubleVector> collector) {
            Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> summaries =
                    (Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector>)
                            getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);

            int expectedNumOfFeatures = (int) summaries.f3.size();
            DenseIntDoubleVector sumVector = Vectors.dense(expectedNumOfFeatures);
            for (Tuple2<Vector, Double> featuresAndLabel : iterable) {
                Preconditions.checkArgument(
                        featuresAndLabel.f0.size() == expectedNumOfFeatures,
                        "Input %s features, but FValueTest is expecting %s features.",
                        featuresAndLabel.f0.size(),
                        expectedNumOfFeatures);

                double yDiff = featuresAndLabel.f1 - summaries.f1;
                Vector<Integer, Double, int[], double[]> features = featuresAndLabel.f0;
                if (yDiff != 0) {
                    double[] values = sumVector.getValues();
                    for (int i = 0; i < expectedNumOfFeatures; i++) {
                        values[i] += yDiff * (features.get(i) - summaries.f3.get(i));
                    }
                }
            }
            BLAS.scal(1.0 / (summaries.f0 - 1), sumVector);
            collector.collect(sumVector);
        }
    }

    /** Computes the p-value, fValues and the number of degrees of freedom of input features. */
    private static class CalFValueOperator
            extends RichMapPartitionFunction<
                    DenseIntDoubleVector, Tuple4<Integer, Double, Long, Double>> {

        private final String broadcastKey;
        private DenseIntDoubleVector sumVector;

        private CalFValueOperator(String broadcastKey) {
            this.broadcastKey = broadcastKey;
        }

        @Override
        public void mapPartition(
                Iterable<DenseIntDoubleVector> iterable,
                Collector<Tuple4<Integer, Double, Long, Double>> collector) {
            Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> summaries =
                    (Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector>)
                            getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
            int expectedNumOfFeatures = (int) summaries.f4.size();

            if (iterable.iterator().hasNext()) {
                sumVector = iterable.iterator().next();
            }
            Preconditions.checkArgument(
                    sumVector.size() == expectedNumOfFeatures,
                    "Input %s features, but FValueTest is expecting %s features.",
                    sumVector.size(),
                    expectedNumOfFeatures);

            final long numSamples = summaries.f0;
            final long degreesOfFreedom = numSamples - 2;

            FDistribution fDistribution = new FDistribution(1, degreesOfFreedom);
            for (int i = 0; i < expectedNumOfFeatures; i++) {
                double covariance = sumVector.get(i);
                double corr = covariance / (summaries.f2 * summaries.f4.get(i));
                double fValue = corr * corr / (1 - corr * corr) * degreesOfFreedom;
                double pValue = 1.0 - fDistribution.cumulativeProbability(fValue);
                collector.collect(Tuple4.of(i, pValue, degreesOfFreedom, fValue));
            }
        }
    }

    /** Computes the num, mean, and standard deviation of the input label and features. */
    private static class SummaryAggregator
            implements AggregateFunction<
                    Tuple2<Vector, Double>,
                    Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector>,
                    Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector>> {

        @Override
        public Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector>
                createAccumulator() {
            return Tuple5.of(0L, 0.0, 0.0, Vectors.dense(), Vectors.dense());
        }

        @Override
        public Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> add(
                Tuple2<Vector, Double> featuresAndLabel,
                Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> summary) {
            Vector<Integer, Double, int[], double[]> features = featuresAndLabel.f0;
            double label = featuresAndLabel.f1;

            if (summary.f0 == 0) {
                summary.f3 = Vectors.dense(features.size());
                summary.f4 = Vectors.dense(features.size());
            }
            summary.f0 += 1L;
            summary.f1 += label;
            summary.f2 += label * label;

            BLAS.axpy(1.0, features, summary.f3);
            double[] summaryF4Values = summary.f4.getValues();
            for (int i = 0; i < features.size(); i++) {
                double v = features.get(i);
                summaryF4Values[i] += v * v;
            }
            return summary;
        }

        @Override
        public Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> getResult(
                Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> summary) {
            final long numRows = summary.f0;
            Preconditions.checkState(numRows > 0, "The training set is empty.");
            int numOfFeatures = (int) summary.f3.size();

            double labelMean = summary.f1 / numRows;
            Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> result =
                    Tuple5.of(
                            numRows,
                            labelMean,
                            Math.sqrt(
                                    (summary.f2 / numRows - labelMean * labelMean)
                                            * numRows
                                            / (numRows - 1)),
                            Vectors.dense(numOfFeatures),
                            Vectors.dense(numOfFeatures));
            for (int i = 0; i < summary.f3.size(); i++) {
                double mean = summary.f3.get(i) / numRows;
                result.f3.set(i, mean);
                result.f4.set(
                        i,
                        Math.sqrt(
                                (summary.f4.get(i) / numRows - mean * mean)
                                        * numRows
                                        / (numRows - 1)));
            }
            return result;
        }

        @Override
        public Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> merge(
                Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> summary1,
                Tuple5<Long, Double, Double, DenseIntDoubleVector, DenseIntDoubleVector> summary2) {
            if (summary1.f0 == 0) {
                return summary2;
            }
            if (summary2.f0 == 0) {
                return summary1;
            }
            summary2.f0 += summary1.f0;
            summary2.f1 += summary1.f1;
            summary2.f2 += summary1.f2;
            BLAS.axpy(1, summary1.f3, summary2.f3);
            BLAS.axpy(1, summary1.f4, summary2.f4);
            return summary2;
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static FValueTest load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
