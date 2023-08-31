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

package org.apache.flink.ml.anomalydetection.isolationforest;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.operator.OperatorStateUtils;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.iteration.ForwardInputsOfLastRound;
import org.apache.flink.ml.common.iteration.TerminateOnMaxIter;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * An Estimator which implements the Isolation Forest algorithm.
 *
 * <p>See https://en.wikipedia.org/wiki/Isolation_forest.
 */
public class IsolationForest
        implements Estimator<IsolationForest, IsolationForestModel>,
                IsolationForestParams<IsolationForest> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public IsolationForest() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public IsolationForestModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Integer treesNumber = getTreesNumber();
        Integer iters = getIters();
        Preconditions.checkArgument(
                treesNumber != null || treesNumber > 0, "Param treesNumber is illegal.");
        Preconditions.checkArgument(iters != null || iters > 0, "Param iters is illegal.");
        IForest iForest = new IForest(treesNumber, iters);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<DenseVector[]> points =
                tEnv.toDataStream(inputs[0]).map(new FormatDataMapFunction(getFeaturesCol()));

        DataStream<IForest> initModelData =
                selectRandomSample1(points).map(new InitModelData(iForest)).setParallelism(1);

        DataStream<IsolationForestModelData> finalModelData =
                Iterations.iterateBoundedStreamsUntilTermination(
                                DataStreamList.of(initModelData),
                                ReplayableDataStreamList.notReplay(points),
                                IterationConfig.newBuilder()
                                        .setOperatorLifeCycle(
                                                IterationConfig.OperatorLifeCycle.ALL_ROUND)
                                        .build(),
                                new IsolationForestIterationBody(iters))
                        .get(0);

        Table finalModelDataTable = tEnv.fromDataStream(finalModelData);
        IsolationForestModel model = new IsolationForestModel().setModelData(finalModelDataTable);
        ParamUtils.updateExistingParams(model, paramMap);
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static IsolationForestModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class FormatDataMapFunction implements MapFunction<Row, DenseVector[]> {
        private final String featuresCol;
        private List<DenseVector> list;

        public FormatDataMapFunction(String featuresCol) {
            this.featuresCol = featuresCol;
        }

        @Override
        public DenseVector[] map(Row row) throws Exception {
            list = new ArrayList<>(256);
            DenseVector denseVector = ((Vector) row.getField(featuresCol)).toDense();
            list.add(denseVector);
            return list.toArray(new DenseVector[0]);
        }
    }

    private static DataStream<DenseVector[]> selectRandomSample1(
            DataStream<DenseVector[]> samplesData) {
        DataStream<DenseVector[]> resultStream =
                DataStreamUtils.mapPartition(
                        DataStreamUtils.sample(samplesData, 256, System.currentTimeMillis()),
                        (MapPartitionFunction<DenseVector[], DenseVector[]>)
                                (iterable, collector) -> {
                                    Iterator<DenseVector[]> samplesDataIterator =
                                            iterable.iterator();
                                    List<DenseVector> list = new ArrayList<>();
                                    while (samplesDataIterator.hasNext()) {
                                        list.addAll(Arrays.asList(samplesDataIterator.next()));
                                    }
                                    collector.collect(list.toArray(new DenseVector[0]));
                                },
                        Types.OBJECT_ARRAY(DenseVectorTypeInfo.INSTANCE));
        resultStream.getTransformation().setParallelism(1);
        return resultStream;
    }

    private static class InitModelData implements MapFunction<DenseVector[], IForest> {
        private final IForest iForest;

        private InitModelData(IForest iForest) {
            this.iForest = iForest;
        }

        @Override
        public IForest map(DenseVector[] denseVectors) throws Exception {
            iForest.createIForest(denseVectors);
            DenseVector scores = iForest.calculateAnomalyScore(denseVectors);
            iForest.classifyByCluster(scores);
            return iForest;
        }
    }

    private static class IsolationForestIterationBody implements IterationBody {
        private final Integer iters;

        public IsolationForestIterationBody(Integer iters) {
            this.iters = iters;
        }

        @Override
        public IterationBodyResult process(
                DataStreamList variableStreams, DataStreamList dataStreams) {
            DataStream<IForest> centersData = variableStreams.get(0);
            DataStream<DenseVector[]> samplesData = dataStreams.get(0);
            final OutputTag<IForest> modelDataOutputTag =
                    new OutputTag<IForest>("IsolationForest") {};

            SingleOutputStreamOperator terminationCriteria =
                    centersData.flatMap(new TerminateOnMaxIter(iters));

            DataStream<IForest> centers =
                    samplesData
                            .connect(centersData.broadcast())
                            .transform(
                                    "CentersUpdateAccumulator",
                                    TypeInformation.of(IForest.class),
                                    new CentersUpdateAccumulator(modelDataOutputTag));

            DataStream<IsolationForestModelData> newModelData =
                    centers.countWindowAll(centers.getParallelism())
                            .reduce(
                                    new ReduceFunction<IForest>() {
                                        @Override
                                        public IForest reduce(IForest iForest1, IForest iForest2)
                                                throws Exception {
                                            if (iForest2.center0 == null
                                                    || iForest2.center1 == null) {
                                                return iForest1;
                                            }
                                            return iForest2;
                                        }
                                    })
                            .flatMap(
                                    new FlatMapFunction<IForest, IsolationForestModelData>() {
                                        @Override
                                        public void flatMap(
                                                IForest iForest,
                                                Collector<IsolationForestModelData> collector)
                                                throws Exception {
                                            if (iForest.center0 != null
                                                    && iForest.center1 != null) {
                                                collector.collect(
                                                        new IsolationForestModelData(iForest));
                                            }
                                        }
                                    });

            DataStream<IForest> newCenters = newModelData.map(x -> x.iForest).setParallelism(1);

            DataStream<IsolationForestModelData> finalModelData =
                    newModelData.flatMap(new ForwardInputsOfLastRound<>());

            return new IterationBodyResult(
                    DataStreamList.of(newCenters),
                    DataStreamList.of(finalModelData),
                    terminationCriteria);
        }
    }

    private static class CentersUpdateAccumulator extends AbstractStreamOperator<IForest>
            implements TwoInputStreamOperator<DenseVector[], IForest, IForest>,
                    IterationListener<IForest> {
        private final OutputTag<IForest> modelDataOutputTag;

        private ListStateWithCache<DenseVector[]> samplesData;

        private ListState<IForest> samplesDataCenter;

        private ListStateWithCache<DenseVector[]> samplesDataScores;

        public CentersUpdateAccumulator(OutputTag<IForest> modelDataOutputTag) {
            this.modelDataOutputTag = modelDataOutputTag;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            samplesDataCenter =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<IForest>(
                                            "centers", TypeInformation.of(IForest.class)));

            samplesData =
                    new ListStateWithCache<>(
                            Types.OBJECT_ARRAY(DenseVectorTypeInfo.INSTANCE)
                                    .createSerializer(getExecutionConfig()),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());

            samplesDataScores =
                    new ListStateWithCache<>(
                            Types.OBJECT_ARRAY(DenseVectorTypeInfo.INSTANCE)
                                    .createSerializer(getExecutionConfig()),
                            getContainingTask(),
                            getRuntimeContext(),
                            context,
                            config.getOperatorID());
        }

        @Override
        public void snapshotState(StateSnapshotContext context) throws Exception {
            super.snapshotState(context);
            samplesData.snapshotState(context);
        }

        @Override
        public void processElement1(StreamRecord<DenseVector[]> streamRecord) throws Exception {
            samplesData.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<IForest> streamRecord) throws Exception {
            Preconditions.checkState(!samplesDataCenter.get().iterator().hasNext());
            samplesDataCenter.add(streamRecord.getValue());
        }

        @Override
        public void onEpochWatermarkIncremented(
                int epochWatermark, Context context, Collector<IForest> collector)
                throws Exception {
            IForest centers =
                    Objects.requireNonNull(
                            OperatorStateUtils.getUniqueElement(samplesDataCenter, "centers")
                                    .orElse(null));
            Iterator<DenseVector[]> samplesDataIterator = samplesData.get().iterator();
            List<DenseVector> list = new ArrayList<>();
            while (samplesDataIterator.hasNext()) {
                DenseVector[] sampleData = samplesDataIterator.next();
                list.add(centers.calculateAnomalyScore(sampleData));
            }
            DenseVector[] scores = list.toArray(new DenseVector[0]);
            samplesDataScores.add(scores);

            collector.collect(samplesDataCenter.get().iterator().next());
            samplesDataCenter.clear();
        }

        @Override
        public void onIterationTerminated(Context context, Collector<IForest> collector)
                throws Exception {
            IForest centers =
                    Objects.requireNonNull(
                            OperatorStateUtils.getUniqueElement(samplesDataCenter, "centers")
                                    .orElse(null));
            double centers0Sum1 = 0.0, centers1Sum1 = 0.0, centers0Sum2 = 0.0, centers1Sum2 = 0.0;
            int size1 = 0, size2 = 0;
            Iterator<DenseVector[]> samplesDataScoresIterator = samplesDataScores.get().iterator();
            while (samplesDataScoresIterator.hasNext()) {
                for (DenseVector denseVector : samplesDataScoresIterator.next()) {
                    DenseVector denseVector1 = centers.classifyByCluster(denseVector);
                    centers0Sum1 += denseVector1.get(0);
                    centers1Sum1 += denseVector1.get(1);
                    size1++;
                }
                centers0Sum2 = centers0Sum1 / size1;
                centers1Sum2 = centers1Sum1 / size1;
                size2++;
            }

            centers.center0 = centers0Sum2 / size2;
            centers.center1 = centers1Sum2 / size2;

            context.output(modelDataOutputTag, centers);

            samplesDataCenter.clear();
            samplesDataScores.clear();
            samplesData.clear();
        }
    }

    /** Construct isolation forest. */
    public static class IForest implements Serializable {
        public int treesNumber;
        public int iters;
        public List<ITree> iTreeList;
        public Double center0;
        public Double center1;
        public int subSamplesSize;

        public IForest() {}

        public IForest(int treesNumber, int iters) {
            this.iters = iters;
            this.treesNumber = treesNumber;
            this.iTreeList = new ArrayList<>();
            this.center0 = null;
            this.center1 = null;
        }

        private void createIForest(DenseVector[] samplesData) throws Exception {
            this.subSamplesSize = Math.min(256, samplesData.length);

            // 限制高度(向上取整)
            int limitHeight = (int) Math.ceil(Math.log(subSamplesSize) / Math.log(2));

            int rows = samplesData.length;

            Random random = new Random(System.currentTimeMillis());
            for (int i = 0; i < this.treesNumber; i++) {
                DenseVector[] subSamples = new DenseVector[subSamplesSize];
                for (int j = 0; j < subSamplesSize; j++) {
                    int r = random.nextInt(rows);
                    subSamples[j] = samplesData[r];
                }
                ITree iTree = ITree.createITree(subSamples, 0, limitHeight);
                this.iTreeList.add(iTree);
            }
        }

        private DenseVector calculateAnomalyScore(DenseVector[] samplesData) throws Exception {
            int n = samplesData.length;

            DenseVector scores = new DenseVector(n);
            for (int i = 0; i < n; i++) {
                double pathLengthSum = 0;
                for (ITree iTree : iTreeList) {
                    pathLengthSum += calculatePathLength(samplesData[i], iTree);
                }

                double pathLengthAvg = pathLengthSum / iTreeList.size();
                double cn = calculateCn(subSamplesSize);
                double index = pathLengthAvg / cn;
                scores.set(i, Math.pow(2, -index));
            }
            return scores;
        }

        private double calculatePathLength(DenseVector sampleData, ITree iTree) throws Exception {
            double pathLength = -1;
            ITree tmpITree = iTree;
            while (tmpITree != null) {
                pathLength += 1;
                if (tmpITree.leftTree == null
                        || tmpITree.rightTree == null
                        || sampleData.get(tmpITree.attributeIndex)
                                == tmpITree.splitAttributeValue) {
                    break;
                } else if (sampleData.get(tmpITree.attributeIndex) < tmpITree.splitAttributeValue) {
                    tmpITree = tmpITree.leftTree;
                } else {
                    tmpITree = tmpITree.rightTree;
                }
            }

            return pathLength + calculateCn(tmpITree.leafNodesNum);
        }

        private double calculateCn(double n) {
            if (n <= 1) {
                return 0;
            }
            return 2.0 * (Math.log(n - 1.0) + 0.5772156649015329) - 2.0 * (n - 1.0) / n;
        }

        private DenseVector classifyByCluster(DenseVector scores) {
            int scoresSize = scores.size();
            this.center0 = scores.get(0); // Cluster center of abnormal
            this.center1 = scores.get(0); // Cluster center of normal

            for (int p = 1; p < scores.size(); p++) {
                if (scores.get(p) > center0) {
                    center0 = scores.get(p);
                }

                if (scores.get(p) < center1) {
                    center1 = scores.get(p);
                }
            }

            int cnt0, cnt1;
            double diff0, diff1;
            int[] labels = new int[scoresSize];

            for (int i = 0; i < iters; i++) {
                cnt0 = 0;
                cnt1 = 0;

                for (int j = 0; j < scoresSize; j++) {
                    diff0 = Math.abs(scores.get(j) - center0);
                    diff1 = Math.abs(scores.get(j) - center1);

                    if (diff0 < diff1) {
                        labels[j] = 0;
                        cnt0++;
                    } else {
                        labels[j] = 1;
                        cnt1++;
                    }
                }

                diff0 = center0;
                diff1 = center1;

                center0 = 0.0;
                center1 = 0.0;
                for (int k = 0; k < scoresSize; k++) {
                    if (labels[k] == 0) {
                        center0 += scores.get(k);
                    } else {
                        center1 += scores.get(k);
                    }
                }

                center0 /= cnt0;
                center1 /= cnt1;

                if (center0 - diff0 <= 1e-6 && center1 - diff1 <= 1e-6) {
                    break;
                }
            }
            return new DenseVector(new double[] {center0, center1});
        }
    }

    /** Construct isolation tree. */
    public static class ITree implements Serializable {
        public int attributeIndex;
        public double splitAttributeValue;
        public ITree leftTree, rightTree;
        public int currentHeight;
        public int leafNodesNum;

        public ITree() {}

        public ITree(int attributeIndex, double splitAttributeValue) {
            this.attributeIndex = attributeIndex;
            this.splitAttributeValue = splitAttributeValue;
            this.leftTree = null;
            this.rightTree = null;
            this.currentHeight = 0;
            this.leafNodesNum = 1;
        }

        public static ITree createITree(
                DenseVector[] samplesData, int currentHeight, int limitHeight) {
            ITree iTree = null;
            if (samplesData.length == 0) {
                return iTree;
            } else if (samplesData.length == 1 || currentHeight >= limitHeight) {
                iTree = new ITree(0, samplesData[0].get(0));
                iTree.leafNodesNum = samplesData.length;
                iTree.currentHeight = currentHeight;
                return iTree;
            }

            int rows = samplesData.length;
            int cols = samplesData[0].size();

            boolean flag = true;
            for (int i = 1; i < rows; i++) {
                if (!samplesData[i].equals(samplesData[i - 1])) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                iTree = new ITree(0, samplesData[0].get(0));
                iTree.leafNodesNum = samplesData.length;
                iTree.currentHeight = currentHeight;
                return iTree;
            }

            Random random = new Random(System.currentTimeMillis());
            int attributeIndex = random.nextInt(cols);

            double maxValue = samplesData[0].get(attributeIndex);
            double minValue = samplesData[0].get(attributeIndex);
            for (int i = 1; i < rows; i++) {
                if (samplesData[i].get(attributeIndex) < minValue) {
                    minValue = samplesData[i].get(attributeIndex);
                }
                if (samplesData[i].get(attributeIndex) > maxValue) {
                    maxValue = samplesData[i].get(attributeIndex);
                }
            }

            double splitAttributeValue = (maxValue - minValue) * random.nextDouble() + minValue;

            int leftNodesNum = 0;
            int rightNodesNum = 0;
            for (int i = 0; i < rows; i++) {
                if (samplesData[i].get(attributeIndex) < splitAttributeValue) {
                    leftNodesNum++;
                } else {
                    rightNodesNum++;
                }
            }

            DenseVector[] leftSamples = new DenseVector[leftNodesNum];
            DenseVector[] rightSamples = new DenseVector[rightNodesNum];
            int l = 0, r = 0;
            for (int i = 0; i < rows; i++) {
                if (samplesData[i].get(attributeIndex) < splitAttributeValue) {
                    leftSamples[l++] = samplesData[i];
                } else {
                    rightSamples[r++] = samplesData[i];
                }
            }

            ITree root = new ITree(attributeIndex, splitAttributeValue);
            root.currentHeight = currentHeight;
            root.leafNodesNum = samplesData.length;
            root.leftTree = createITree(leftSamples, currentHeight + 1, limitHeight);
            root.rightTree = createITree(rightSamples, currentHeight + 1, limitHeight);

            return root;
        }
    }
}
