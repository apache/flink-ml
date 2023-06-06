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

package org.apache.flink.ml.evaluation.binaryclassification;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.tuple.Tuple4;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.linalg.IntDoubleVector;
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
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * An AlgoOperator which calculates the evaluation metrics for binary classification. The input data
 * has columns rawPrediction, label and an optional weight column. The rawPrediction can be of type
 * double (binary 0/1 prediction, or probability of label 1) or of type vector (length-2 vector of
 * raw predictions, scores, or label probabilities). The output may contain different metrics which
 * will be defined by parameter MetricsNames. See {@link BinaryClassificationEvaluatorParams}.
 */
public class BinaryClassificationEvaluator
        implements AlgoOperator<BinaryClassificationEvaluator>,
                BinaryClassificationEvaluatorParams<BinaryClassificationEvaluator> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private static final int NUM_SAMPLE_FOR_RANGE_PARTITION = 100;
    private static final Logger LOG = LoggerFactory.getLogger(BinaryClassificationEvaluator.class);

    public BinaryClassificationEvaluator() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Tuple3<Double, Boolean, Double>> evalData =
                tEnv.toDataStream(inputs[0])
                        .map(new ParseSample(getLabelCol(), getRawPredictionCol(), getWeightCol()));
        final String boundaryRangeKey = "boundaryRange";
        final String partitionSummariesKey = "partitionSummaries";

        DataStream<Tuple4<Double, Boolean, Double, Integer>> evalDataWithTaskId =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(evalData),
                        Collections.singletonMap(boundaryRangeKey, getBoundaryRange(evalData)),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(new AppendTaskId(boundaryRangeKey));
                        });

        /* Repartition the evaluated data by range. */
        evalDataWithTaskId =
                evalDataWithTaskId.partitionCustom((chunkId, numPartitions) -> chunkId, x -> x.f3);

        /* Sorts local data by score.*/
        DataStream<Tuple3<Double, Boolean, Double>> sortEvalData =
                DataStreamUtils.mapPartition(
                        evalDataWithTaskId,
                        new MapPartitionFunction<
                                Tuple4<Double, Boolean, Double, Integer>,
                                Tuple3<Double, Boolean, Double>>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple4<Double, Boolean, Double, Integer>> values,
                                    Collector<Tuple3<Double, Boolean, Double>> out) {
                                List<Tuple3<Double, Boolean, Double>> bufferedData =
                                        new LinkedList<>();
                                for (Tuple4<Double, Boolean, Double, Integer> t4 : values) {
                                    bufferedData.add(Tuple3.of(t4.f0, t4.f1, t4.f2));
                                }
                                bufferedData.sort(Comparator.comparingDouble(o -> -o.f0));
                                for (Tuple3<Double, Boolean, Double> dataPoint : bufferedData) {
                                    out.collect(dataPoint);
                                }
                            }
                        });

        /* Calculates the summary of local data. */
        DataStream<BinarySummary> partitionSummaries =
                sortEvalData.transform(
                        "reduceInEachPartition",
                        TypeInformation.of(BinarySummary.class),
                        new PartitionSummaryOperator());

        /* Sorts global data. Output Tuple4 : <score, order, isPositive, weight>. */
        DataStream<Tuple4<Double, Long, Boolean, Double>> dataWithOrders =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(sortEvalData),
                        Collections.singletonMap(partitionSummariesKey, partitionSummaries),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.flatMap(new CalcSampleOrders(partitionSummariesKey));
                        });

        DataStream<double[]> localAreaUnderROCVariable =
                dataWithOrders.transform(
                        "AccumulateMultiScore",
                        TypeInformation.of(double[].class),
                        new AccumulateMultiScoreOperator());

        DataStream<double[]> middleAreaUnderROC =
                DataStreamUtils.reduce(
                        localAreaUnderROCVariable,
                        (ReduceFunction<double[]>)
                                (t1, t2) -> {
                                    t2[0] += t1[0];
                                    t2[1] += t1[1];
                                    t2[2] += t1[2];
                                    return t2;
                                });

        DataStream<Double> areaUnderROC =
                middleAreaUnderROC.map(
                        (MapFunction<double[], Double>)
                                value -> {
                                    if (value[1] > 0 && value[2] > 0) {
                                        return (value[0] - 1. * value[1] * (value[1] + 1) / 2)
                                                / (value[1] * value[2]);
                                    } else {
                                        return Double.NaN;
                                    }
                                });

        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        broadcastMap.put(partitionSummariesKey, partitionSummaries);
        broadcastMap.put(AREA_UNDER_ROC, areaUnderROC);
        DataStream<BinaryMetrics> localMetrics =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(sortEvalData),
                        broadcastMap,
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return DataStreamUtils.mapPartition(
                                    input, new CalcBinaryMetrics(partitionSummariesKey));
                        });

        DataStream<Map<String, Double>> metrics =
                DataStreamUtils.mapPartition(
                        localMetrics, new MergeMetrics(), Types.MAP(Types.STRING, Types.DOUBLE));
        metrics.getTransformation().setParallelism(1);

        final String[] metricsNames = getMetricsNames();
        TypeInformation<?>[] metricTypes = new TypeInformation[metricsNames.length];
        Arrays.fill(metricTypes, Types.DOUBLE);
        RowTypeInfo outputTypeInfo = new RowTypeInfo(metricTypes, metricsNames);

        DataStream<Row> evalResult =
                metrics.map(
                        (MapFunction<Map<String, Double>, Row>)
                                value -> {
                                    Row ret = new Row(metricsNames.length);
                                    for (int i = 0; i < metricsNames.length; ++i) {
                                        ret.setField(i, value.get(metricsNames[i]));
                                    }
                                    return ret;
                                },
                        outputTypeInfo);
        return new Table[] {tEnv.fromDataStream(evalResult)};
    }

    /** Updates variables for calculating AreaUnderROC. */
    private static class AccumulateMultiScoreOperator extends AbstractStreamOperator<double[]>
            implements OneInputStreamOperator<Tuple4<Double, Long, Boolean, Double>, double[]>,
                    BoundedOneInput {
        private ListState<double[]> accValueState;
        private ListState<Double> scoreState;

        double[] accValue;
        double score;

        @Override
        public void endInput() {
            if (accValue != null) {
                output.collect(
                        new StreamRecord<>(
                                new double[] {
                                    accValue[0] / accValue[1] * accValue[2],
                                    accValue[2],
                                    accValue[3]
                                }));
            }
        }

        @Override
        public void processElement(
                StreamRecord<Tuple4<Double, Long, Boolean, Double>> streamRecord) {
            Tuple4<Double, Long, Boolean, Double> t = streamRecord.getValue();
            if (accValue == null) {
                accValue = new double[4];
                score = t.f0;
            } else if (score != t.f0) {
                output.collect(
                        new StreamRecord<>(
                                new double[] {
                                    accValue[0] / accValue[1] * accValue[2],
                                    accValue[2],
                                    accValue[3]
                                }));
                Arrays.fill(accValue, 0.0);
            }
            accValue[0] += t.f1;
            accValue[1] += 1.0;
            if (t.f2) {
                accValue[2] += t.f3;
            } else {
                accValue[3] += t.f3;
            }
        }

        @Override
        @SuppressWarnings("unchecked")
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            accValueState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "accValueState", TypeInformation.of(double[].class)));
            accValue =
                    OperatorStateUtils.getUniqueElement(accValueState, "accValueState")
                            .orElse(null);

            scoreState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "scoreState", TypeInformation.of(Double.class)));
            score = OperatorStateUtils.getUniqueElement(scoreState, "scoreState").orElse(0.0);
        }

        @Override
        @SuppressWarnings("unchecked")
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            accValueState.clear();
            scoreState.clear();
            if (accValue != null) {
                accValueState.add(accValue);
                scoreState.add(score);
            }
        }
    }

    private static class PartitionSummaryOperator extends AbstractStreamOperator<BinarySummary>
            implements OneInputStreamOperator<Tuple3<Double, Boolean, Double>, BinarySummary>,
                    BoundedOneInput {
        private ListState<BinarySummary> summaryState;
        private BinarySummary summary;

        @Override
        public void endInput() {
            if (summary != null) {
                output.collect(new StreamRecord<>(summary));
            }
        }

        @Override
        public void processElement(StreamRecord<Tuple3<Double, Boolean, Double>> streamRecord) {
            updateBinarySummary(summary, streamRecord.getValue());
        }

        @Override
        @SuppressWarnings("unchecked")
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            summaryState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "summaryState",
                                            TypeInformation.of(BinarySummary.class)));
            summary =
                    OperatorStateUtils.getUniqueElement(summaryState, "summaryState")
                            .orElse(
                                    new BinarySummary(
                                            getRuntimeContext().getIndexOfThisSubtask(),
                                            -Double.MAX_VALUE,
                                            0,
                                            0));
        }

        @Override
        @SuppressWarnings("unchecked")
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            summaryState.clear();
            if (summary != null) {
                summaryState.add(summary);
            }
        }
    }

    /** Merges the metrics calculated locally and output metrics data. */
    private static class MergeMetrics
            implements MapPartitionFunction<BinaryMetrics, Map<String, Double>> {
        @Override
        public void mapPartition(
                Iterable<BinaryMetrics> values, Collector<Map<String, Double>> out) {
            Iterator<BinaryMetrics> iter = values.iterator();
            BinaryMetrics reduceMetrics = iter.next();
            while (iter.hasNext()) {
                reduceMetrics = reduceMetrics.merge(iter.next());
            }
            Map<String, Double> map = new HashMap<>();
            map.put(AREA_UNDER_ROC, reduceMetrics.areaUnderROC);
            map.put(AREA_UNDER_PR, reduceMetrics.areaUnderPR);
            map.put(AREA_UNDER_LORENZ, reduceMetrics.areaUnderLorenz);
            map.put(KS, reduceMetrics.ks);
            out.collect(map);
        }
    }

    private static class CalcBinaryMetrics
            extends RichMapPartitionFunction<Tuple3<Double, Boolean, Double>, BinaryMetrics> {
        private final String partitionSummariesKey;

        public CalcBinaryMetrics(String partitionSummariesKey) {
            this.partitionSummariesKey = partitionSummariesKey;
        }

        @Override
        public void mapPartition(
                Iterable<Tuple3<Double, Boolean, Double>> iterable,
                Collector<BinaryMetrics> collector) {

            List<BinarySummary> statistics =
                    getRuntimeContext().getBroadcastVariable(partitionSummariesKey);
            long[] countValues =
                    reduceBinarySummary(statistics, getRuntimeContext().getIndexOfThisSubtask());

            double areaUnderROC =
                    getRuntimeContext().<Double>getBroadcastVariable(AREA_UNDER_ROC).get(0);
            long totalTrue = countValues[2];
            long totalFalse = countValues[3];
            if (totalTrue == 0) {
                LOG.warn("There is no positive sample in data!");
            }
            if (totalFalse == 0) {
                LOG.warn("There is no negative sample in data!");
            }

            BinaryMetrics metrics = new BinaryMetrics(0L, areaUnderROC);
            double[] tprFprPrecision = new double[4];
            for (Tuple3<Double, Boolean, Double> t3 : iterable) {
                updateBinaryMetrics(t3, metrics, countValues, tprFprPrecision);
            }
            collector.collect(metrics);
        }
    }

    private static void updateBinaryMetrics(
            Tuple3<Double, Boolean, Double> cur,
            BinaryMetrics binaryMetrics,
            long[] countValues,
            double[] recordValues) {
        if (binaryMetrics.count == 0) {
            recordValues[0] = countValues[2] == 0 ? 1.0 : 1.0 * countValues[0] / countValues[2];
            recordValues[1] = countValues[3] == 0 ? 1.0 : 1.0 * countValues[1] / countValues[3];
            recordValues[2] =
                    countValues[0] + countValues[1] == 0
                            ? 1.0
                            : 1.0 * countValues[0] / (countValues[0] + countValues[1]);
            recordValues[3] =
                    1.0 * (countValues[0] + countValues[1]) / (countValues[2] + countValues[3]);
        }

        binaryMetrics.count++;
        if (cur.f1) {
            countValues[0]++;
        } else {
            countValues[1]++;
        }

        double tpr = countValues[2] == 0 ? 1.0 : 1.0 * countValues[0] / countValues[2];
        double fpr = countValues[3] == 0 ? 1.0 : 1.0 * countValues[1] / countValues[3];
        double precision =
                countValues[0] + countValues[1] == 0
                        ? 1.0
                        : 1.0 * countValues[0] / (countValues[0] + countValues[1]);
        double positiveRate =
                1.0 * (countValues[0] + countValues[1]) / (countValues[2] + countValues[3]);

        binaryMetrics.areaUnderLorenz +=
                ((positiveRate - recordValues[3]) * (tpr + recordValues[0]) / 2);
        binaryMetrics.areaUnderPR += ((tpr - recordValues[0]) * (precision + recordValues[2]) / 2);
        binaryMetrics.ks = Math.max(Math.abs(fpr - tpr), binaryMetrics.ks);

        recordValues[0] = tpr;
        recordValues[1] = fpr;
        recordValues[2] = precision;
        recordValues[3] = positiveRate;
    }

    /**
     * For each sample, calculates its score order among all samples. The sample with minimum score
     * has order 1, while the sample with maximum score has order samples.
     *
     * <p>Input is a dataset of tuple (score, is real positive, weight), output is a dataset of
     * tuple (score, order, is real positive, weight).
     */
    private static class CalcSampleOrders
            extends RichFlatMapFunction<
                    Tuple3<Double, Boolean, Double>, Tuple4<Double, Long, Boolean, Double>> {
        private long startIndex;
        private long total = -1;
        private final String partitionSummariesKey;

        public CalcSampleOrders(String partitionSummariesKey) {
            this.partitionSummariesKey = partitionSummariesKey;
        }

        @Override
        public void flatMap(
                Tuple3<Double, Boolean, Double> value,
                Collector<Tuple4<Double, Long, Boolean, Double>> out)
                throws Exception {
            if (total == -1) {
                List<BinarySummary> statistics =
                        getRuntimeContext().getBroadcastVariable(partitionSummariesKey);
                long[] countValues =
                        reduceBinarySummary(
                                statistics, getRuntimeContext().getIndexOfThisSubtask());
                startIndex = countValues[1] + countValues[0] + 1;
                total = countValues[2] + countValues[3];
            }
            out.collect(Tuple4.of(value.f0, total - startIndex + 1, value.f1, value.f2));
            startIndex++;
        }
    }

    /**
     * @param values Reduce Summary of all workers.
     * @param taskId current taskId.
     * @return [curTrue, curFalse, TotalTrue, TotalFalse]
     */
    private static long[] reduceBinarySummary(List<BinarySummary> values, int taskId) {
        List<BinarySummary> list = new ArrayList<>(values);
        list.sort(Comparator.comparingDouble(t -> -t.maxScore));
        long curTrue = 0;
        long curFalse = 0;
        long totalTrue = 0;
        long totalFalse = 0;

        for (BinarySummary statistics : list) {
            if (statistics.taskId == taskId) {
                curFalse = totalFalse;
                curTrue = totalTrue;
            }
            totalTrue += statistics.curPositive;
            totalFalse += statistics.curNegative;
        }
        return new long[] {curTrue, curFalse, totalTrue, totalFalse};
    }

    /**
     * Updates binary summary by one evaluated element.
     *
     * @param statistics Binary summary.
     * @param evalElement evaluated element.
     */
    private static void updateBinarySummary(
            BinarySummary statistics, Tuple3<Double, Boolean, Double> evalElement) {
        if (evalElement.f1) {
            statistics.curPositive++;
        } else {
            statistics.curNegative++;
        }
        if (Double.compare(statistics.maxScore, evalElement.f0) < 0) {
            statistics.maxScore = evalElement.f0;
        }
    }

    /**
     * Appends taskId for every sample as range defined. If sample score between range[i] and
     * range[i+1], taskId is i.
     */
    private static class AppendTaskId
            extends RichMapFunction<
                    Tuple3<Double, Boolean, Double>, Tuple4<Double, Boolean, Double, Integer>> {
        private double[] boundaryRange;
        private final String boundaryRangeKey;

        public AppendTaskId(String boundaryRangeKey) {
            this.boundaryRangeKey = boundaryRangeKey;
        }

        @Override
        public Tuple4<Double, Boolean, Double, Integer> map(Tuple3<Double, Boolean, Double> value)
                throws Exception {
            if (boundaryRange == null) {
                boundaryRange =
                        (double[])
                                getRuntimeContext().getBroadcastVariable(boundaryRangeKey).get(0);
            }
            for (int i = boundaryRange.length - 1; i > 0; --i) {
                if (value.f0 > boundaryRange[i]) {
                    return Tuple4.of(value.f0, value.f1, value.f2, i);
                }
            }
            return Tuple4.of(value.f0, value.f1, value.f2, 0);
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static BinaryClassificationEvaluator load(StreamTableEnvironment env, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Calculates boundary range for rangePartition.
     *
     * @param evalData Evaluate data.
     * @return Boundary range.
     */
    private static DataStream<double[]> getBoundaryRange(
            DataStream<Tuple3<Double, Boolean, Double>> evalData) {
        DataStream<double[]> sampleScoreStream =
                DataStreamUtils.mapPartition(
                        evalData,
                        new MapPartitionFunction<Tuple3<Double, Boolean, Double>, double[]>() {
                            @Override
                            public void mapPartition(
                                    Iterable<Tuple3<Double, Boolean, Double>> dataPoints,
                                    Collector<double[]> out) {
                                List<Double> bufferedDataPoints = new ArrayList<>();
                                for (Tuple3<Double, Boolean, Double> dataPoint : dataPoints) {
                                    bufferedDataPoints.add(dataPoint.f0);
                                }
                                double[] sampleScores = new double[NUM_SAMPLE_FOR_RANGE_PARTITION];
                                Arrays.fill(sampleScores, Double.MAX_VALUE);
                                Random rand = new Random();
                                int sampleNum = bufferedDataPoints.size();
                                if (sampleNum > 0) {
                                    for (int i = 0; i < NUM_SAMPLE_FOR_RANGE_PARTITION; ++i) {
                                        sampleScores[i] =
                                                bufferedDataPoints.get(rand.nextInt(sampleNum));
                                    }
                                }
                                out.collect(sampleScores);
                            }
                        });
        final int parallel = sampleScoreStream.getParallelism();

        DataStream<double[]> boundaryRange =
                DataStreamUtils.mapPartition(
                        sampleScoreStream,
                        new MapPartitionFunction<double[], double[]>() {
                            @Override
                            public void mapPartition(
                                    Iterable<double[]> dataPoints, Collector<double[]> out) {
                                double[] allSampleScore =
                                        new double[parallel * NUM_SAMPLE_FOR_RANGE_PARTITION];
                                int cnt = 0;
                                for (double[] dataPoint : dataPoints) {
                                    System.arraycopy(
                                            dataPoint,
                                            0,
                                            allSampleScore,
                                            cnt * NUM_SAMPLE_FOR_RANGE_PARTITION,
                                            NUM_SAMPLE_FOR_RANGE_PARTITION);
                                    cnt++;
                                }
                                Arrays.sort(allSampleScore);
                                double[] boundaryRange = new double[parallel];
                                for (int i = 0; i < parallel; ++i) {
                                    boundaryRange[i] =
                                            allSampleScore[i * NUM_SAMPLE_FOR_RANGE_PARTITION];
                                }
                                out.collect(boundaryRange);
                            }
                        });
        boundaryRange.getTransformation().setParallelism(1);
        return boundaryRange;
    }

    private static class ParseSample implements MapFunction<Row, Tuple3<Double, Boolean, Double>> {
        private final String labelCol;
        private final String rawPredictionCol;
        private final String weightCol;

        public ParseSample(String labelCol, String rawPredictionCol, String weightCol) {
            this.labelCol = labelCol;
            this.rawPredictionCol = rawPredictionCol;
            this.weightCol = weightCol;
        }

        @Override
        public Tuple3<Double, Boolean, Double> map(Row value) throws Exception {
            double label = ((Number) value.getFieldAs(labelCol)).doubleValue();
            Object probOrigin = value.getField(rawPredictionCol);
            double prob =
                    probOrigin instanceof IntDoubleVector
                            ? ((IntDoubleVector) probOrigin).get(1)
                            : ((Number) probOrigin).doubleValue();
            double weight =
                    weightCol == null ? 1.0 : ((Number) value.getField(weightCol)).doubleValue();
            return Tuple3.of(prob, label == 1.0, weight);
        }
    }

    /** Binary Summary of data in one worker. */
    public static class BinarySummary implements Serializable {
        public Integer taskId;
        // maximum score in this partition
        public double maxScore;
        // real positives in this partition
        public long curPositive;
        // real negatives in this partition
        public long curNegative;

        public BinarySummary() {}

        public BinarySummary(Integer taskId, double maxScore, long curPositive, long curNegative) {
            this.taskId = taskId;
            this.maxScore = maxScore;
            this.curPositive = curPositive;
            this.curNegative = curNegative;
        }
    }

    /** The evaluation metrics for binary classification. */
    public static class BinaryMetrics {
        /* The count of samples. */
        public long count;

        /* Area under ROC */
        public double areaUnderROC;

        /* Area under Lorenz */
        public double areaUnderLorenz;

        /* Area under PRC */
        public double areaUnderPR;

        /* KS */
        public double ks;

        public BinaryMetrics() {}

        public BinaryMetrics(long count, double areaUnderROC) {
            this.count = count;
            this.areaUnderROC = areaUnderROC;
        }

        public BinaryMetrics merge(BinaryMetrics binaryClassMetrics) {
            if (null == binaryClassMetrics) {
                return this;
            }
            Preconditions.checkState(
                    Double.compare(areaUnderROC, binaryClassMetrics.areaUnderROC) == 0,
                    "AreaUnderROC not equal!");
            count += binaryClassMetrics.count;
            ks = Math.max(ks, binaryClassMetrics.ks);
            areaUnderPR += binaryClassMetrics.areaUnderPR;
            areaUnderLorenz += binaryClassMetrics.areaUnderLorenz;
            return this;
        }
    }
}
