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

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.distance.DistanceMeasure;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.VectorWithNorm;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * OnlineKMeansModel can be regarded as an advanced {@link KMeansModel} operator which can update
 * model data in a streaming format, using the model data provided by {@link OnlineKMeans}.
 */
public class OnlineKMeansModel
        implements Model<OnlineKMeansModel>, KMeansModelParams<OnlineKMeansModel> {
    public static final String MODEL_DATA_VERSION_GAUGE_KEY = "modelDataVersion";

    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public OnlineKMeansModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public OnlineKMeansModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), Types.INT),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol()));

        DataStream<Row> predictionResult =
                tEnv.toDataStream(inputs[0])
                        .connect(KMeansModelData.getModelDataStream(modelDataTable).broadcast())
                        .transform(
                                "PredictLabelOperator",
                                outputTypeInfo,
                                new PredictLabelOperator(
                                        inputTypeInfo,
                                        getFeaturesCol(),
                                        DistanceMeasure.getInstance(getDistanceMeasure()),
                                        getK()));

        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    /** A utility operator used for prediction. */
    private static class PredictLabelOperator extends AbstractStreamOperator<Row>
            implements TwoInputStreamOperator<Row, KMeansModelData, Row> {
        private final RowTypeInfo inputTypeInfo;

        private final String featuresCol;

        private final DistanceMeasure distanceMeasure;

        private final int k;

        private VectorWithNorm[] centroids;

        // TODO: replace this with a complete solution of reading first model data from unbounded
        // model data stream before processing the first predict data.
        private ListState<Row> bufferedPointsState;

        /**
         * Basic implementation of the model data version with the following rules.
         *
         * <ul>
         *   <li>Negative value is regarded as illegal value.
         *   <li>Zero value means the version has not been initialized yet.
         *   <li>Positive value represents valid version.
         * </ul>
         */
        // TODO: replace this simple implementation of model data version with the formal API to
        // track model version after its design is settled.
        private int modelDataVersion = 0;

        public PredictLabelOperator(
                RowTypeInfo inputTypeInfo,
                String featuresCol,
                DistanceMeasure distanceMeasure,
                int k) {
            this.inputTypeInfo = inputTypeInfo;
            this.featuresCol = featuresCol;
            this.distanceMeasure = distanceMeasure;
            this.k = k;
        }

        @Override
        public void initializeState(StateInitializationContext context) throws Exception {
            super.initializeState(context);

            bufferedPointsState =
                    context.getOperatorStateStore()
                            .getListState(
                                    new ListStateDescriptor<>("bufferedPoints", inputTypeInfo));
        }

        @Override
        public void open() throws Exception {
            super.open();

            getRuntimeContext()
                    .getMetricGroup()
                    .gauge(
                            MODEL_DATA_VERSION_GAUGE_KEY,
                            (Gauge<String>) () -> Integer.toString(modelDataVersion));
        }

        @Override
        public void processElement1(StreamRecord<Row> streamRecord) throws Exception {
            Row dataPoint = streamRecord.getValue();
            if (centroids == null) {
                bufferedPointsState.add(dataPoint);
                return;
            }
            DenseVector point = (DenseVector) ((Vector) dataPoint.getField(featuresCol)).toDense();
            int closestCentroidId =
                    distanceMeasure.findClosest(centroids, new VectorWithNorm(point));
            output.collect(new StreamRecord<>(Row.join(dataPoint, Row.of(closestCentroidId))));
        }

        @Override
        public void processElement2(StreamRecord<KMeansModelData> streamRecord) throws Exception {
            KMeansModelData modelData = streamRecord.getValue();
            Preconditions.checkArgument(modelData.centroids.length <= k);
            centroids = new VectorWithNorm[modelData.centroids.length];
            for (int i = 0; i < centroids.length; i++) {
                centroids[i] = new VectorWithNorm(modelData.centroids[i]);
            }
            modelDataVersion++;
            for (Row dataPoint : bufferedPointsState.get()) {
                processElement1(new StreamRecord<>(dataPoint));
            }
            bufferedPointsState.clear();
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Saves the metadata to the given path.
     *
     * <p>NOTE: the unbounded model data table will not be saved. Model data needs be explicitly
     * exported with {@link OnlineKMeansModel#getModelData()}.
     */
    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    // TODO: Add INFO level logging.
    public static OnlineKMeansModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
