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
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.iteration.DataStreamList;
import org.apache.flink.iteration.IterationBody;
import org.apache.flink.iteration.IterationBodyResult;
import org.apache.flink.iteration.IterationConfig;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.iteration.Iterations;
import org.apache.flink.iteration.ReplayableDataStreamList;
import org.apache.flink.iteration.datacache.nonkeyed.ListStateWithCache;
import org.apache.flink.iteration.datacache.nonkeyed.OperatorScopeManagedMemoryManager;
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
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

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

        IForest iForest = new IForest(getNumTrees());
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        DataStream<DenseVector[]> points =
                tEnv.toDataStream(inputs[0]).map(new FormatDataMapFunction(getFeaturesCol()));

        DataStream<IForest> initModelData =
                selectRandomSample(points, getMaxSamples())
                        .map(new InitModelData(iForest, getMaxIter(), getMaxFeatures()))
                        .setParallelism(1);

        DataStream<IsolationForestModelData> finalModelData =
                Iterations.iterateBoundedStreamsUntilTermination(
                                DataStreamList.of(initModelData),
                                ReplayableDataStreamList.notReplay(points),
                                IterationConfig.newBuilder()
                                        .setOperatorLifeCycle(
                                                IterationConfig.OperatorLifeCycle.ALL_ROUND)
                                        .build(),
                                new IsolationForestIterationBody(getMaxIter()))
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

    public static IsolationForest load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    private static class FormatDataMapFunction implements MapFunction<Row, DenseVector[]> {
        private final String featuresCol;

        public FormatDataMapFunction(String featuresCol) {
            this.featuresCol = featuresCol;
        }

        @Override
        public DenseVector[] map(Row row) throws Exception {
            List<DenseVector> list = new ArrayList<>(256);
            DenseVector denseVector = ((Vector) row.getField(featuresCol)).toDense();
            list.add(denseVector);
            return list.toArray(new DenseVector[0]);
        }
    }

    private static DataStream<DenseVector[]> selectRandomSample(
            DataStream<DenseVector[]> samplesData, int maxSamples) {
        DataStream<DenseVector[]> resultStream =
                DataStreamUtils.mapPartition(
                        DataStreamUtils.sample(samplesData, maxSamples, System.nanoTime()),
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

    private static class InitModelData extends RichMapFunction<DenseVector[], IForest> {
        private final IForest iForest;
        private final int iters;
        private final double maxFeatures;

        private InitModelData(IForest iForest, int iters, double maxFeatures) {
            this.iForest = iForest;
            this.iters = iters;
            this.maxFeatures = maxFeatures;
        }

        @Override
        public IForest map(DenseVector[] denseVectors) throws Exception {
            int n = denseVectors[0].size();
            int numFeatures = Math.min(n, Math.max(1, (int) (maxFeatures * n)));

            List<Integer> tempList = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                tempList.add(i);
            }
            Collections.shuffle(tempList);

            int[] featuresIndicts = new int[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                featuresIndicts[j] = tempList.get(j);
            }

            iForest.generateIsolationForest(denseVectors, featuresIndicts);
            DenseVector scores = iForest.calculateScore(denseVectors);
            iForest.classifyByCluster(scores, iters);
            return iForest;
        }
    }

    private static class IsolationForestIterationBody implements IterationBody {
        private final int iters;

        public IsolationForestIterationBody(int iters) {
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
                                    new CentersUpdateAccumulator(modelDataOutputTag, iters));

            DataStreamUtils.setManagedMemoryWeight(centers, 100);

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

        private final int iters;

        private ListStateWithCache<DenseVector[]> samplesData;

        private ListState<IForest> samplesDataCenter;

        private ListStateWithCache<DenseVector[]> samplesDataScores;

        public CentersUpdateAccumulator(OutputTag<IForest> modelDataOutputTag, int iters) {
            this.modelDataOutputTag = modelDataOutputTag;
            this.iters = iters;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            samplesDataCenter =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>(
                                            "centers", TypeInformation.of(IForest.class)));

            TypeInformation<DenseVector[]> type =
                    ObjectArrayTypeInfo.getInfoFor(DenseVectorTypeInfo.INSTANCE);

            final StreamTask<?, ?> containingTask = getContainingTask();
            final OperatorID operatorID = config.getOperatorID();
            final OperatorScopeManagedMemoryManager manager =
                    OperatorScopeManagedMemoryManager.getOrCreate(containingTask, operatorID);
            final String samplesDataStateKey = "data-state";
            final String samplesDataScoresStateKey = "scores-state";
            manager.register(samplesDataStateKey, 1.);
            manager.register(samplesDataScoresStateKey, 1.);

            samplesData =
                    new ListStateWithCache<>(
                            type.createSerializer(getExecutionConfig()),
                            samplesDataStateKey,
                            context,
                            this);

            samplesDataScores =
                    new ListStateWithCache<>(
                            type.createSerializer(getExecutionConfig()),
                            samplesDataScoresStateKey,
                            context,
                            this);
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
                list.add(centers.calculateScore(sampleData));
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
            double centers0Sum1 = 0.0;
            double centers1Sum1 = 0.0;
            double centers0Sum2 = 0.0;
            double centers1Sum2 = 0.0;
            int size1 = 0;
            int size2 = 0;
            Iterator<DenseVector[]> samplesDataScoresIterator = samplesDataScores.get().iterator();
            while (samplesDataScoresIterator.hasNext()) {
                for (DenseVector denseVector : samplesDataScoresIterator.next()) {
                    DenseVector denseVector1 = centers.classifyByCluster(denseVector, iters);
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
}
