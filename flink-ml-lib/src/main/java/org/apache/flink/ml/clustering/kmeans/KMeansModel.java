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

package org.apache.flink.ml.clustering.kmeans;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.api.java.typeutils.ObjectArrayTypeInfo;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.distance.DistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.OnCheckpointRollingPolicy;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A Model which clusters data into k clusters using the model data computed by {@link KMeans}. */
public class KMeansModel implements Model<KMeansModel>, KMeansModelParams<KMeansModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table centroidsTable;

    public KMeansModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public KMeansModel setModelData(Table... inputs) {
        centroidsTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {centroidsTable};
    }

    @Override
    public Table[] transform(Table... inputs) {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<DenseVector[]> centroids =
                tEnv.toDataStream(centroidsTable).map(row -> (DenseVector[]) row.getField("f0"));

        String featureCol = getFeaturesCol();
        String predictionCol = getPredictionCol();
        DistanceMeasure distanceMeasure = DistanceMeasure.getInstance(getDistanceMeasure());

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), Types.INT),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), predictionCol));

        DataStream<Row> input = tEnv.toDataStream(inputs[0]);
        DataStream<Row> output =
                input.connect(centroids.broadcast())
                        .transform(
                                "SelectNearestCentroid",
                                outputTypeInfo,
                                new SelectNearestCentroidOperator(featureCol, distanceMeasure));

        return new Table[] {tEnv.fromDataStream(output)};
    }

    private static class SelectNearestCentroidOperator extends AbstractStreamOperator<Row>
            implements TwoInputStreamOperator<Row, DenseVector[], Row> {
        private ListState<Row> inputs;
        private ListState<DenseVector[]> centroids;

        private final String featureCol;
        private final DistanceMeasure distanceMeasure;

        public SelectNearestCentroidOperator(String featureCol, DistanceMeasure distanceMeasure) {
            this.featureCol = featureCol;
            this.distanceMeasure = distanceMeasure;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);
            inputs =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("points", Row.class));
            TypeInformation<DenseVector[]> type =
                    ObjectArrayTypeInfo.getInfoFor(DenseVectorTypeInfo.INSTANCE);
            centroids =
                    context.getOperatorStateStore()
                            .getListState(new ListStateDescriptor<>("centroids", type));
        }

        @Override
        public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
            inputs.add(streamRecord.getValue());
        }

        @Override
        public void processElement2(StreamRecord<DenseVector[]> streamRecord) throws Exception {
            centroids.add(streamRecord.getValue());
        }

        @Override
        public void finish() throws Exception {
            List<DenseVector[]> list = IteratorUtils.toList(centroids.get().iterator());
            if (list.size() != 1) {
                throw new RuntimeException(
                        "The operator received "
                                + list.size()
                                + " list of centroids in this round");
            }
            DenseVector[] centroidValues = list.get(0);

            for (Row input : inputs.get()) {
                DenseVector point = (DenseVector) input.getField(featureCol);

                double minDistance = Double.MAX_VALUE;
                int closestCentroidId = -1;

                for (int i = 0; i < centroidValues.length; i++) {
                    DenseVector centroid = centroidValues[i];
                    double distance = distanceMeasure.distance(centroid, point);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestCentroidId = i;
                    }
                }

                output.collect(new StreamRecord<>(Row.join(input, Row.of(closestCentroidId))));
            }
            inputs.clear();
            centroids.clear();
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) centroidsTable).getTableEnvironment();

        String dataPath = ReadWriteUtils.getDataPath(path);
        FileSink<DenseVector[]> sink =
                FileSink.forRowFormat(new Path(dataPath), new KMeansModelData.ModelDataEncoder())
                        .withRollingPolicy(OnCheckpointRollingPolicy.build())
                        .withBucketAssigner(new BasePathBucketAssigner<>())
                        .build();
        tEnv.toDataStream(centroidsTable)
                .map(row -> (DenseVector[]) row.getField("f0"))
                .sinkTo(sink);

        ReadWriteUtils.saveMetadata(this, path);
    }

    // TODO: Add INFO level logging.
    public static KMeansModel load(StreamExecutionEnvironment env, String path) throws IOException {
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        Source<DenseVector[], ?, ?> source =
                FileSource.forRecordStreamFormat(
                                new KMeansModelData.ModelDataStreamFormat(),
                                ReadWriteUtils.getDataPaths(path))
                        .build();
        KMeansModel model = ReadWriteUtils.loadStageParam(path);
        DataStream<DenseVector[]> modelData =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), "modelData");
        return model.setModelData(tEnv.fromDataStream(modelData, KMeansModelData.SCHEMA));
    }
}
